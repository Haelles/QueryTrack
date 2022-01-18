import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
import pdb


@HEADS.register_module
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     with_proj=True,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_tracking=dict(
                     type='FocalLoss',
                     use_softmax=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 with_avg_pool=False,
                 num_fcs=2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True
                 ):
        super(TrackHead, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        # else:
        #    in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                           if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)  # 2层： (256*7*7, 1024) (1024, 1024)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic = dynamic

        self.fp16_enabled = False
        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.tracking_loss = build_loss(loss_tracking)

    def init_weights(self):
        # TODO 暂时设定为DynamicConv部分采用QueryInst的初始化方法、fc部分采用原任务提出者的初始化方法
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                        device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                                     device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                   torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                   + self.match_coeff[2] * label_delta

    @auto_fp16()
    def forward(self, x, ref_x, proposal_feat, ref_proposal_feat, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching

        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        ref_proposal_feat = ref_proposal_feat.reshape(-1, self.in_channels)
        x = self.instance_interactive_conv(
            proposal_feat, x)  # (b*n, 256)
        ref_x = self.instance_interactive_conv(
            ref_proposal_feat, ref_x)  # (b*n, 256)

        assert len(x_n) == len(ref_x_n)  # batch_size
        if self.with_avg_pool:  # False
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)  # num_all_proposals, 256*7*7
        ref_x = ref_x.view(ref_x.size(0), -1)
        
        # print("--------------")
        # print(x.shape)
        # print(ref_x.shape)
        # print("in tracking forward")
        for idx, fc in enumerate(self.fcs):  # 2层： (256*7*7, 1024) (1024, 1024)
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)

        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        # print("current num: {n}".format(n=n))
        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            # print("current prod {prod}".format(prod=prod))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = prod.new_zeros((m, 1))

                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=torch.cuda.current_device())
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy, prods_all], dim=2)
        return match_score

    def loss(self,
             match_score,
             ids,
             reduction_override=None):
        # TODO 检查avg_factor及其它
        losses = dict()

        n = 0  # batch_size
        x_n = [s.size(0) for s in match_score]
        
        loss_match = 0.
        match_acc = 0.
        n_total = 0
        # print("loss track_head.py")
        for score, cur_ids in zip(match_score, ids):
            # pos_inds = ids > 0
            # print("score {score}".format(score=score.shape))
            # print("cur_ids {cur_ids}".format(cur_ids=cur_ids))
            # pdb.set_trace()    
            num_samples = cur_ids.size(0)
            if num_samples == 0:
                loss_match += (score.sum() * 0.0)
            else:
                n += 1
                id_weights = score.new_ones(num_samples)
                # id_weights[pos_inds] = 1.0
                num_pos = score.new_ones(num_samples).float().sum()
                avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
                loss_match += self.tracking_loss(
                                        score,
                                        cur_ids,  # (b*n, ) 对应的ref gt labels
                                        id_weights,  # (b*n, )
                                        avg_factor=avg_factor,
                                        reduction_override=reduction_override)
                n_total += num_samples
                match_acc += accuracy(score, cur_ids)
        # TODO 是否需要改一下这个n，之前continue那些是否需要考虑
        losses['loss_track'] = loss_match / n
        losses['match_acc'] = match_acc / n_total
        return losses
