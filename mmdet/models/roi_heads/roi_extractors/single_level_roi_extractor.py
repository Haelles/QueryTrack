import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        # rois: (batch * num_proposals, 5)
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()  # 64位整型
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        # feats: list 比如[(B, 256, 200, 200), (B, 256, 100, 100), (B, 256, 50, 50), (B, 256, 25, 25)]
        # rois: (batch * num_proposals, 5) 实际:确实是torch.Size([200, 5])
        out_size = self.roi_layers[0].output_size  # bbox部分对应固定的(7, 7) mask_head对应(14, 14) MaskTrack也是这样
        num_levels = len(feats)  # == 4
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])  # (-1, 256*7*7==12544)
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:  # default == false 不执行
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)
        # rois: (batch * num_proposals, 5)
        # target_lvls (batch * num_proposals, ) 看起来是一个0-3的index 一维张量序列
        # 实际确实是torch.Size([200]) 根据scale映射到4个不同层
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:  # default is None 不执行
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):  # 4
            # mask:
            mask = target_lvls == i  # 获得当前level需要用到的region proposal坐标
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1).expand(*expand_dims).reshape(
                    roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois)
                roi_feats_t *= mask
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)  # 去掉多余的维度，变成一维的
            if inds.numel() > 0:  # 返回数组中元素的个数
                rois_ = rois[inds]
                # rois_ (n, 5) n为这个level要用到的region proposal坐标数
                # feats[i] e.g. (B, 256, 200, 200) 这个level的特征（之前从FPN中提取到的）
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                # roi_feats_t (n, default_num_proposals==256, 7, 7) 或是mask的(n, 256, 14, 14)
                roi_feats[inds] = roi_feats_t  # 通过布尔数组赋值过去
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        # 应该是(batch * num_proposals, 256, 7, 7) ; dict: output_size=7
        # 打印确实是torch.Size([200, 256, 7, 7])，各个roi_feats_t赋值回去
        return roi_feats
