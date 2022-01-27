import torch
import torch.nn as nn
from mmcv import imresize
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer, ConvModule, Conv2d, build_upsample_layer)
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.utils import build_transformer

from .fcn_mask_head import FCNMaskHead
import numpy as np


@HEADS.register_module()
class DynamicMaskHead(FCNMaskHead):
    
    def __init__(self,
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=14,
                     with_proj=False,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 num_classes=80,
                 dropout=0.,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(type='DiceLoss', loss_weight=8.0),
                 **kwargs):
        super(DynamicMaskHead, self).__init__(
            num_convs=num_convs,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            conv_kernel_size=conv_kernel_size,
            conv_out_channels=conv_out_channels,
            num_classes=num_classes,
            class_agnostic=class_agnostic,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            loss_mask=loss_mask)
        assert class_agnostic is False, "DynamicMaskHead only support class_agnostic=False"
        self.fp16_enabled = False

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """

        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)  # (b*n, 256)

        x = proposal_feat_iic.permute(0, 2, 1).reshape(roi_feat.size())

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        num_pos = labels.new_ones(labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            loss_mask = self.loss_mask(mask_pred[torch.arange(num_pos).long(), labels, ...].sigmoid(),
                                       mask_targets,
                                       avg_factor=avg_factor)
        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self,
                    sampling_results,
                    gt_masks,
                    rcnn_train_cfg):
        
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    # def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
    #                   ori_shape, scale_factor, rescale, det_obj_ids=None):
    #     """Get segmentation masks from mask_pred and bboxes.

    #     Args:
    #         mask_pred (Tensor or ndarray): shape (n, #class, h, w).
    #             For single-scale testing, mask_pred is the direct output of
    #             model, whose type is Tensor, while for multi-scale testing,
    #             it will be converted to numpy array outside of this method.
    #         det_bboxes (Tensor): shape (n, 4/5)
    #         det_labels (Tensor): shape (n, )
    #         img_shape (Tensor): shape (3, )
    #         rcnn_test_cfg (dict): rcnn testing config
    #         ori_shape: original image size

    #     Returns:
    #         list[list]: encoded masks
    #     """
    #     if isinstance(mask_pred, torch.Tensor):
    #         mask_pred = mask_pred.sigmoid().cpu().numpy()
    #     assert isinstance(mask_pred, np.ndarray)

    #     cls_segms = [[] for _ in range(self.num_classes)]  # TODO TO CHECK 原任务这里为self.num_classes - 1
    #     if det_obj_ids is not None:
    #         obj_segms = {}
    #     bboxes = det_bboxes.cpu().numpy()[:, :4]
    #     labels = det_labels.cpu().numpy()  # TODO TO CHECK + 1
    #     scale_factor = scale_factor.cpu().numpy()

    #     if rescale:
    #         img_h, img_w = ori_shape[:2]
    #     else:
    #         img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
    #         img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
    #         scale_factor = 1.0

    #     for i in range(bboxes.shape[0]):
    #         bbox = (bboxes[i, :] / scale_factor).astype(np.int32)  # 还原为原图尺寸
    #         label = labels[i]
    #         w = max(bbox[2] - bbox[0] + 1, 1)
    #         h = max(bbox[3] - bbox[1] + 1, 1)

    #         if not self.class_agnostic:  # True
    #             mask_pred_ = mask_pred[i, label, :, :]
    #         else:
    #             mask_pred_ = mask_pred[i, 0, :, :]
    #         im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    #         bbox_mask = imresize(mask_pred_, (w, h))  # 将(28,28)的mask分数转变为原图bbox区域的mask分数
    #         bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
    #             np.uint8)  # 原图bbox区域的mask完全生成
    #         im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask  # 构造出当前实例的0-1mask
    #         rle = mask_util.encode(
    #             np.array(im_mask[:, :, np.newaxis], order='F'))[0]
    #         if det_obj_ids is not None:
    #             if det_obj_ids[i] >= 0:
    #                 obj_segms[det_obj_ids[i]] = rle
    #         else:
    #             cls_segms[label].append(rle)
    #     if det_obj_ids is not None:
    #         return obj_segms
    #     else:
    #         return cls_segms