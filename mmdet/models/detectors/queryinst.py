from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch


@DETECTORS.register_module()
class QueryInst(TwoStageDetector):
    r"""Implementation of `QueryInst: Parallelly Supervised Mask Query for
     Instance Segmentation <https://arxiv.org/abs/2105.01928>`, based on 
     SparseRCNN detector. """
    
    def __init__(self, *args, **kwargs):
        super(QueryInst, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'QueryInst do not support external proposals'

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_data=None,
                      gt_pids=None,
                      **kwargs):
        """

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.
            ref_data (None | dict) : reference data.
            gt_pids (list[Tensor]): Reference from key_ann to ref_ann

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'QueryInst does not support' \
                                  ' external proposals'
        assert gt_masks is not None, 'QueryInst needs mask groundtruth annotations' \
                                  ' for instance segmentation'
        # x: list 每个元素为(batch, , , )
        # csdn: ResNet50+FPN输出C2, C3, C4, C5四个分辨率的特征
        # 比如[(B, 256, 200, 200), (B, 256, 100, 100), (B, 256, 50, 50), (B, 256, 25, 25)]。
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_data['img'])
        # proposal_boxes为(batch_size ,num_proposals, 4)
        # proposal_features (batch, num_proposals, proposal_feature_channel)
        # imgs_whwh (batch_size ,num_proposals, 4)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)
        #
        ref_imgs_whwh = []
        for meta in ref_data['img_metas']:
            h, w, _ = meta['img_shape']
            ref_imgs_whwh.append(ref_data['img'][0].new_tensor([[w, h, w, h]]))
        ref_imgs_whwh = torch.cat(ref_imgs_whwh, dim=0)
        ref_imgs_whwh = ref_imgs_whwh[:, None, :]
        ref_data['imgs_whwh'] = ref_imgs_whwh
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh,
            ref_data=ref_data,
            ref_x=ref_x,
            gt_pids=gt_pids)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.aug_test_rpn(x, img_metas)
        results = self.roi_head.aug_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            aug_imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs
