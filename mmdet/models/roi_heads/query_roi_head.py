import torch
import numpy as np

from mmdet.core import bbox2result, bbox2result_with_id,bbox2roi, bbox_xyxy_to_cxcywh, bbox_flip
from mmdet.core.bbox.samplers import PseudoSampler
from mmdet.core import bbox_overlaps
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead

from mmcv.ops.nms import batched_nms


def mask2results(mask_preds, det_labels, num_classes):
    cls_segms = [[] for _ in range(num_classes)]
    for i in range(mask_preds.shape[0]):
        cls_segms[det_labels[i]].append(mask_preds[i])
    return cls_segms


def get_queries(pos_inds, pos_assigned_gt_inds, queries):
    num_ref_gt = pos_inds.size(0)
    len_single_query = queries[0].size(0)
    res = queries.new_zeros([num_ref_gt, len_single_query])
    for (pos_ind, pos_assigned_gt_ind) in zip(pos_inds, pos_assigned_gt_inds):
        res[pos_assigned_gt_ind, :] = queries[pos_ind, :]
    return res


@HEADS.register_module()
class QueryRoIHead(CascadeRoIHead):
    r"""

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        bbox_head (dict): Config of box head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 track_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),  # TODO ?????????????????????size=7?????????????????????
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 bbox_head=dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_head=dict(
                     type='DynamicMaskHead',
                     dynamic_conv_cfg=dict(
                         type='DynamicConv',
                         in_channels=256,
                         feat_channels=64,
                         out_channels=256,
                         input_feat_shape=14,
                         with_proj=False,
                         act_cfg=dict(type='ReLU', inplace=True),
                         norm_cfg=dict(type='LN')),
                     dropout=0.0,
                     num_convs=4,
                     roi_feat_size=14,
                     in_channels=256,
                     conv_kernel_size=3,
                     conv_out_channels=256,
                     class_agnostic=False,
                     norm_cfg=dict(type='BN'),
                     upsample_cfg=dict(type='deconv', scale_factor=2),
                     loss_dice=dict(type='DiceLoss', loss_weight=8.0)),
                 track_head=dict(
                     type='TrackHead',
                     num_fcs=2,
                     in_channels=256,
                     fc_out_channels=1024,
                     roi_feat_size=7,
                     match_coeff=[1.0, 2.0, 10],
                     loss_tracking=dict(
                         type='FocalLoss',
                         use_sigmoid=False,
                         use_softmax=True,
                         gamma=2.0,
                         alpha=0.25,  # TODO ?????????????????????
                         loss_weight=1.0),

                     dynamic_conv_cfg=dict(
                         type='DynamicConv',
                         in_channels=256,
                         feat_channels=64,
                         out_channels=256,
                         input_feat_shape=7,
                         with_proj=True,
                         act_cfg=dict(type='ReLU', inplace=True),
                         norm_cfg=dict(type='LN'))),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert mask_roi_extractor is not None
        assert bbox_head is not None
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super(QueryRoIHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            track_roi_extractor=track_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            track_head=track_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'QueryInst only support `PseudoSampler`'
        
        # for test memory queue
        self.prev_bboxes =  None
        self.prev_roi_feats = None
        self.prev_det_labels = None
        self.prev_attn_feats = None


    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)  # batch_size
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        # TODO ????????????rois???(num_proposal, 5)????????????batch*num_proposal ??????????????????
        # bbox_roi_extractor.num_inputs: ????????? == 4  ( featmap_strides=[4, 8, 16, 32]
        # x: ??????[(B, 256, 200, 200), (B, 256, 100, 100), (B, 256, 50, 50), (B, 256, 25, 25)]
        # csdn??????????????????bbox_feats: (batch * num_proposals, 256, 7, 7)
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)  # ?????????x^{FPN}????????????b_{t-1}

        # bbox_feats: x_{t}^{box}
        # ?????????object_feats: (batch_size, num_proposals, proposal_feature_channel)
        # ?????????????????????object query?????????q_{t-1}
        # ?????????object_feats: (batch_size, num_proposal, feature_dimensions)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats,
                                                                   object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg ?????????"??????/?????????"
            bbox_pred.view(-1, bbox_pred.size(-1)),  # torch.Size([2, 100, 4]) -> (2*100, 4)
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        # proposal torch.Size([100, 4])
        # proposal torch.Size([100, 4])
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),  # (batch*num_proposals, 4)
            object_feats=object_feats,  # q_{t} x_{t}^{box*}
            attn_feats=attn_feats,  # attn_feats: torch.Size([batch, num_proposals, 256])
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        # mask_feats == x_{t}^{mask} torch.Size([34, 256, 14, 14]) torch.Size([8, 256, 14, 14])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)  # x^{FPN}  b_{t}  ?????????out_channel=256?????????dim1???256
        # mask_pred torch.Size([34, 80, 28, 28]) torch.Size([8, 80, 28, 28])
        # mask_roi_extractor???????????????RoIAlign?????????mask_feats??????roi_feat
        # attn_feats?????????queries???????????????????????? --> proposal_feat
        mask_pred = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, attn_feats, sampling_results, gt_masks, rcnn_train_cfg):

        if sum([len(gt_mask) for gt_mask in gt_masks]) == 0:
            print('Ground Truth Not Found!')
            loss_mask = sum([_.sum() for _ in self.mask_head[stage].parameters()]) * 0.
            return dict(loss_mask=loss_mask)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # attn_feats: torch.Size([batch, num_proposals, 256])
        attn_feats = torch.cat([feats[res.pos_inds] for (feats, res) in zip(attn_feats, sampling_results)])
        # mask_results['mask_pred'] == m_{t} torch.Size([34, 80, 28, 28]) torch.Size([8, 80, 28, 28])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)
        # ???gt???bitmap mask?????????
        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def _track_forward_train(self, stage, x, attn_feats, sampling_results, ref_x, ref_sampling_results, ref_bboxes, label):
        # attn_feats: torch.Size([batch, num_proposals, 256])
        if sum([len(sampling_result.pos_inds) for sampling_result in sampling_results]) == 0:
            print('Track Forward: Positive Results Not Found!')
            loss_track = sum([_.sum() for _ in self.track_head[stage].parameters()]) * 0.
            return dict(loss_track=loss_track)
        if sum([len(ref_sampling_result.pos_inds) for ref_sampling_result in ref_sampling_results]) == 0:
            print('Track Forward: Ref Positive Results Not Found!')
            loss_track = sum([_.sum() for _ in self.track_head[stage].parameters()]) * 0.
            return dict(loss_track=loss_track)
        if sum([len(ref_bbox) for ref_bbox in ref_bboxes]) == 0:
            print('Track Forward: Ref_bbox Not Found!')
            loss_track = sum([_.sum() for _ in self.track_head[stage].parameters()]) * 0.
            return dict(loss_track=loss_track)
        
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        bbox_img_n = [res.pos_bboxes.size(0) for res in sampling_results]
        # TODO ???????????????????????????????????????????????????????????????bbox??????roi??????????????????pos_bbox
        # TODO ???????????????assign_result?????????gt_bbox???????????????
        bbox_attn_feats = torch.cat([feats[res.pos_inds] for (feats, res) in zip(attn_feats, sampling_results)])
        # TODO ??????????????????????????????????????????ref_gt_bbox????????????pred_bbox?????????????????????query
        ref_rois = bbox2roi(ref_bboxes)
        ref_bbox_img_n = [res.size(0) for res in ref_bboxes]
        # TODO TO CHECK ??????????????????????????????????????????????????????????????????queries??????????????????
        with torch.no_grad():
            ref_bbox_attn_feats = torch.cat([
                get_queries(ref_sampling_result.pos_inds, ref_sampling_result.pos_assigned_gt_inds, attn_feat)
                for (attn_feat, ref_sampling_result) in zip(attn_feats, ref_sampling_results)
            ])

        track_roi_extractor = self.track_roi_extractor[stage]
        track_head = self.track_head[stage]
        # track_feats == x_{t}^{track}
        track_feats = track_roi_extractor(x[:track_roi_extractor.num_inputs],  # torch.Size([num_all, 256, 7, 7])
                                          pos_rois)  # x^{FPN}  b_{t}
        with torch.no_grad():
            ref_track_feats = track_roi_extractor(ref_x[:track_roi_extractor.num_inputs],
                                                ref_rois)
        match_score = track_head(track_feats, ref_track_feats, bbox_attn_feats, ref_bbox_attn_feats, bbox_img_n, ref_bbox_img_n)
        # match_score: list tensor(len(pos_rois), len(ref_rois))
        loss_track = track_head.loss(match_score, label)

        track_results = loss_track
        return track_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None,
                      ref_data=None,
                      ref_x=None,
                      gt_pids=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features. ???FPN?????????????????????type???tuple ????????????==(batch,channel,h,w)
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            ref_data (None | dict) : reference data.
            ref_x (list[Tensor]): list of multi-level ref_img features
            gt_pids (list[Tensor]): Reference from key_ann to ref_ann

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """
        num_imgs = len(img_metas)  # batch
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)  # (batch, num, 4)
        ref_imgs_whwh = ref_data['imgs_whwh'].repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        # proposal_boxes: (batch_size ,num_proposals, 4)
        # proposal_list???????????????len???batch_size??????????????????(num_proposals, 4)?????????
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]  # len: batch_size
        object_feats = proposal_features  # (batch_size, num_proposals, proposal_feature_channel)
        all_stage_loss = {}

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)  # (batch * num_proposals, 5) 5: img_index, x1, y1, x2, y2
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)  # rois: not requires_grad
            with torch.no_grad():
                # ??????????????????ref frame???q_{t-1}^{*}???b_{t}
                # TODO ?????????????????????????????? ???????????????key frame?????????rois b_{t-1}???object_feats q_{t-1}
                ref_bbox_results = self._bbox_forward(stage, ref_x, rois, object_feats,
                                                      ref_data['img_metas'])
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:  # ??????
                # T support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            ref_sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']  # not requires_grad
            proposal_list = bbox_results['detach_proposal_list']  # not requires_grad
            for i in range(num_imgs):
                normolize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                          imgs_whwh[i])
                # < AssignResult(num_gts=10, gt_inds.shape = (100,), max_overlaps = None, labels.shape = (100,)) >
                # ???proposal_list[i]???????????????????????????????????????????????????????????????gt bbox??????
                assign_result = self.bbox_assigner[stage].assign(
                    normolize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                # < SamplingResult({
                #     'neg_bboxes': torch.Size([90, 4]),
                #     'neg_inds': tensor([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                #                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                #                         38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
                #                         59, 60, 61, 62, 63, 64, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                #                         79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 94, 96, 97, 98, 99],
                #                        device='cuda:0'),
                #     'num_gts': 10,
                #     'pos_assigned_gt_inds': tensor([3, 4, 5, 7, 9, 1, 6, 0, 2, 8], device='cuda:0'),
                #     'pos_bboxes': torch.Size([10, 4]),
                #     'pos_inds': tensor([4, 8, 44, 46, 53, 65, 69, 88, 91, 95], device='cuda:0'),
                #     'pos_is_gt': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.uint8)
                # }) >
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)

                # TODO ???????????????????????????????????????bbox head??????ref frame???roi???cls_pred
                with torch.no_grad():
                    ref_cls_pred_list = ref_bbox_results['detach_cls_score_list']
                    ref_proposal_list = ref_bbox_results['detach_proposal_list']
                    ref_normolize_bbox_ccwh = bbox_xyxy_to_cxcywh(ref_proposal_list[i] /
                                                                  ref_imgs_whwh[i])
                    ref_assign_result = self.bbox_assigner[stage].assign(
                        ref_normolize_bbox_ccwh, ref_cls_pred_list[i], ref_data['gt_bboxes'][i],
                        ref_data['gt_labels'][i], ref_data['img_metas'][i])

                    ref_sampling_result = self.bbox_sampler[stage].sample(
                        ref_assign_result, ref_proposal_list[i], ref_data['gt_bboxes'][i])
                    ref_sampling_results.append(ref_sampling_result)

            # ?????????labels, label_weights, bbox_targets, bbox_weights?????????gt?????????gt bbox?????????
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)
            cls_score = bbox_results['cls_score']  # torch.Size([2, 100, 80]) requires_grad
            decode_bbox_pred = bbox_results['decode_bbox_pred']  # requires_grad
            object_feats = bbox_results['object_feats']  # requires_grad

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)

            if self.with_mask:
                # x: FPN
                # bbox_results['attn_feats']: q_{t-1}^{*}
                # bbox_results['attn_feats'] requires_grad
                mask_results = self._mask_forward_train(stage, x, bbox_results['attn_feats'],
                                                        sampling_results, gt_masks, self.train_cfg[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            if self.with_track:
                ids = []  # ??????label??? sampling_result?????????-1???
                for (gt_pid, sampling_result) in zip(gt_pids, sampling_results):
                    cur_id = gt_pid[sampling_result.pos_assigned_gt_inds]
                    # if cur_id.size(0) == 0:
                    #    import pdb
                    #    pdb.set_trace()
                    #    print("cur_id size == 0 ")
                    #    print("{pos}\n {gt_pid}".format(pos=sampling_result.pos_assigned_gt_inds, gt_pid=gt_pid))
                    ids.append(cur_id)
                track_results = self._track_forward_train(stage, x, bbox_results['attn_feats'], sampling_results, ref_x, ref_sampling_results, ref_data['gt_bboxes'], ids)
                single_stage_loss['loss_track'] = track_results['loss_track']
                # single_stage_loss['loss_track'] = mask_results['loss_mask'].new_ones(1).squeeze(0)
            for key, value in single_stage_loss.items():
                # print(key)
                # print(value)
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                                        self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            bbox_results (list[tuple[np.ndarray]]): \
                [[cls1_det, cls2_det, ...], ...]. \
                The outer list indicates images, and the inner \
                list indicates per-class detected bboxes. The \
                np.ndarray has shape (num_det, 5) and the last \
                dimension 5 represents (x1, y1, x2, y2, score).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        # print(img_metas, flush=True)
        # [{'filename': 'data/coco/val2017/000000397133.jpg', 'ori_filename': '000000397133.jpg',
        #   'ori_shape': (427, 640, 3), 'img_shape': (800, 1199, 3), 'pad_shape': (800, 1216, 3),
        #   'scale_factor': array([1.8734375, 1.8735363, 1.8734375, 1.8735363], dtype=float32), 'flip': False,
        #   'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28, 103.53], dtype=float32),
        #                                            'std': array([58.395, 57.12, 57.375], dtype=float32),
        #                                            'to_rgb': True}, 'batch_input_shape': (800, 1216)}]
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}

        object_feats = proposal_features
        bbox_results = {}
        # import pdb
        # pdb.set_trace()
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']  # QueryInst: [1, 100, 80]
            proposal_list = bbox_results['detach_proposal_list']

        if self.with_mask:
            rois = bbox2roi(proposal_list)
            # ????????????_mask_forward_train()
            mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
            # in QueryInst train: mask_pred torch.Size([100, 80, 28, 28]) / torch.Size([8, 80, 28, 28])
            # ??????????????????????????????
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                num_imgs, -1, *mask_results['mask_pred'].size()[1:]
            )  # (b*n, 80, 28, 28) -> (b, n, 80, 28, 28)

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:  # ???????????????
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            # TODO TO CHECK ??????MaskTrackRCNN????????????0.05???threshold
            topk_indices = topk_indices[scores_per_img > 0.05]
            scores_per_img = scores_per_img[scores_per_img > 0.05]
            labels_per_img = topk_indices % num_classes  # ????????? ????????? ????????????
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]  # ?????? ?????????bbox ????????????
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))  # (10, 4) (10, 1) -> (10, 5)
            det_labels.append(labels_per_img)  # (10, )

        # ??????tracking??????
        det_obj_ids=np.array([], dtype=np.int64)
        is_first = img_metas[0]['is_first']
        # ??????????????????????????????????????????????????????0???
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]  # (n, )
        if det_bboxes.nelement()==0:
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
                self.prev_topk_indices = None
        else:
            res_det_bboxes = det_bboxes.clone()
            if rescale:  # ??????rescale==True ???????????????resize????????????
                scale_factor = res_det_bboxes.new_tensor(img_metas[0]['scale_factor'])
                res_det_bboxes[:, :4] *= scale_factor

            det_rois = bbox2roi([res_det_bboxes])
            det_roi_feats = self.track_roi_extractor[stage](
                x[:self.track_roi_extractor[stage].num_inputs], det_rois)
            # ???????????????bbox??????x_{t}^{track}
            # import pdb
            # pdb.set_trace()
            if is_first or (not is_first and self.prev_bboxes is None):
                # ????????????????????????????????????????????????????????????bbox????????????
                det_obj_ids = np.arange(det_bboxes.size(0))  # (n, ) det_bboxes???????????????
                # save bbox and features for later matching
                self.prev_bboxes = det_bboxes
                self.prev_roi_feats = det_roi_feats  #  ??????(batch * num_proposals, 256, 7, 7)
                self.prev_det_labels = det_labels
                self.prev_attn_feats = torch.cat([bbox_results['attn_feats'][0][topk_indice // num_classes][None] for topk_indice in topk_indices])
            else:
                assert self.prev_roi_feats is not None
                assert self.prev_bboxes is not None
                # only support one image at a time
                bbox_img_n = [det_bboxes.size(0)]
                prev_bbox_img_n = [self.prev_roi_feats.size(0)]

                bbox_attn_feats = torch.cat([bbox_results['attn_feats'][0][topk_indice // num_classes][None] for topk_indice in topk_indices])
                prev_bbox_attn_feats = self.prev_attn_feats

                match_score = self.track_head[stage](det_roi_feats, self.prev_roi_feats,
                                        bbox_attn_feats, prev_bbox_attn_feats,
                                        bbox_img_n, prev_bbox_img_n)[0]
                match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
                label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()  # MaskTrack torch.Size([2, 4])
                # TODO TO CHECK ??????????????????????????????????????????????????????
                bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
                # compute comprehensive score 
                # (m, n) m????????? ?????????n???
                comp_scores = self.track_head[stage].compute_comp_scores(match_logprob, 
                    det_bboxes[:,4].view(-1, 1),  # (n, 4) -> (n*4, 1)
                    bbox_ious,
                    label_delta,
                    add_bbox_dummy=True)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object, 
                # add tracking features/bboxes of new object
                match_ids = match_ids.cpu().numpy().astype(np.int32)
                det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
                best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100.0)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        # ???????????????????????????????????????
                        det_obj_ids[idx] = self.prev_roi_feats.size(0)
                        self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                        self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                        self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                        self.prev_attn_feats = torch.cat((self.prev_attn_feats, bbox_attn_feats[idx][None]), dim=0)
                    else:
                        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
                        obj_id = match_id - 1
                        match_score = comp_scores[idx, match_id]  # ??????????????????????????????????????????
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            # udpate feature
                            self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                            self.prev_bboxes[obj_id] = det_bboxes[idx]
                            self.prev_attn_feats[obj_id] = bbox_attn_feats[idx]
        # import pdb
        # pdb.set_trace()
        # ????????????????????????????????????label/?????????????????? det_bboxes???????????????
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids, num_classes)
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[:, :4] *
                scale_factors[0] if rescale else det_bboxes[:, :4]
            ]  # ???????????????
            det_labels = [det_labels]
            segm_results = []
            mask_pred = mask_results['mask_pred']  # (b, n, 80, 28, 28)
            for img_id in range(num_imgs):
                # ??????query/pred bbox??????num_classes???mask
                # mask_pred[img_id] (n, 80, 28, 28) -> (n*80, 28, 28)
                # cls_score[img_id] (n, 80)
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                # (10, 28, 28) -> (10, 1, 28, 28) -> (10, 80, 28, 28)
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                    self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                    rescale, det_obj_ids=det_obj_ids)  # ???????????????????????????dict
                segm_results.append(segm_result)

            ms_segm_result['ensemble'] = segm_results[0]

        if self.with_mask:
            return ms_bbox_result['ensemble'], ms_segm_result['ensemble']
        else:
            return ms_bbox_result['ensemble']

    def aug_test(self,
                 aug_x,
                 aug_proposal_boxes,
                 aug_proposal_features,
                 aug_img_metas,
                 aug_imgs_whwh,
                 rescale=False):

        samples_per_gpu = len(aug_img_metas[0])
        aug_det_bboxes = [[] for _ in range(samples_per_gpu)]
        aug_det_labels = [[] for _ in range(samples_per_gpu)]
        aug_mask_preds = [[] for _ in range(samples_per_gpu)]
        for x, proposal_boxes, proposal_features, img_metas, imgs_whwh in \
                zip(aug_x, aug_proposal_boxes, aug_proposal_features, aug_img_metas, aug_imgs_whwh):

            num_imgs = len(img_metas)
            proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
            ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            object_feats = proposal_features
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)
                object_feats = bbox_results['object_feats']
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list']

            if self.with_mask:
                rois = bbox2roi(proposal_list)
                mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
                mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                    num_imgs, -1, *mask_results['mask_pred'].size()[1:]
                )

            num_classes = self.bbox_head[-1].num_classes
            det_bboxes = []
            det_labels = []

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = cls_score.softmax(-1)[..., :-1]

            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
                labels_per_img = topk_indices % num_classes
                bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                          num_classes]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                aug_det_bboxes[img_id].append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                aug_det_labels[img_id].append(labels_per_img)
                det_labels.append(labels_per_img)

            if self.with_mask:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_pred = mask_results['mask_pred']
                for img_id in range(num_imgs):
                    mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                    mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                    segm_result = self.mask_head[-1].get_seg_masks(
                        mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                        self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                        rescale, format=False)
                    aug_mask_preds[img_id].append(segm_result.detach().cpu().numpy())

        det_bboxes, det_labels, mask_preds = [], [], []

        for img_id in range(samples_per_gpu):
            for aug_id in range(len(aug_det_bboxes[img_id])):
                img_meta = aug_img_metas[aug_id][img_id]
                img_shape = img_meta['ori_shape']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                aug_det_bboxes[img_id][aug_id][:, :-1] = bbox_flip(aug_det_bboxes[img_id][aug_id][:, :-1],
                                                                   img_shape, flip_direction) if flip else \
                    aug_det_bboxes[img_id][aug_id][:, :-1]
                if flip:
                    if flip_direction == 'horizontal':
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, :, ::-1]
                    else:
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, ::-1, :]

        for img_id in range(samples_per_gpu):
            det_bboxes_per_im = torch.cat(aug_det_bboxes[img_id])
            det_labels_per_im = torch.cat(aug_det_labels[img_id])
            mask_preds_per_im = np.concatenate(aug_mask_preds[img_id])

            # TODO(vealocia): implement batched_nms here.
            det_bboxes_per_im, keep_inds = batched_nms(det_bboxes_per_im[:, :-1], det_bboxes_per_im[:, -1].contiguous(),
                                                       det_labels_per_im, self.test_cfg.nms)
            det_bboxes_per_im = det_bboxes_per_im[:self.test_cfg.max_per_img, ...]
            det_labels_per_im = det_labels_per_im[keep_inds][:self.test_cfg.max_per_img, ...]
            mask_preds_per_im = mask_preds_per_im[keep_inds.detach().cpu().numpy()][:self.test_cfg.max_per_img, ...]
            det_bboxes.append(det_bboxes_per_im)
            det_labels.append(det_labels_per_im)
            mask_preds.append(mask_preds_per_im)

        ms_bbox_result = {}
        ms_segm_result = {}
        num_classes = self.bbox_head[-1].num_classes
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(samples_per_gpu)
        ]
        ms_bbox_result['ensemble'] = bbox_results
        mask_results = [
            mask2results(mask_preds[i], det_labels[i], num_classes)
            for i in range(samples_per_gpu)
        ]
        ms_segm_result['ensemble'] = mask_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)

                all_stage_bbox_results.append(bbox_results)
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
        return all_stage_bbox_results
