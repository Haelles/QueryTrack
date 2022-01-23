import os.path as osp
import warnings
from collections import OrderedDict

import torch

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

import os.path as osp
import random

from .custom import CustomDataset
from pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module()
class YTVISDataset(CustomDataset):
    # TODO 后续调整这些类的名字
    CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan',
               'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow', 'fish',
               'train', 'horse', 'turtle', 'bear', 'motorbike', 'giraffe', 'leopard',
               'fox', 'deer', 'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
               'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal',
               'tennis_racket')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 pipeline,
                 seg_prefix=None,
                 img_scale=None,
                 img_norm_cfg=None,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)  # "videos"
        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))  # type == tuple
        self.img_ids = img_ids  # 一系列元组，由在视频的编号idx和在视频内部唯一的编号frame_id组成

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                          if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        # cfg: img_scale=(640, 360) 转变成list
        # self.img_scales = img_scale if isinstance(img_scale,
        #                                           list) else [img_scale]
        # assert mmcv.is_list_of(self.img_scales, tuple)

        # normalization configs
        # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        # self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio  # 0.5
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor  # == 32

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param  # None
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()  # Set flag according to image aspect ratio 设置横纵比
        # transforms
        self.pipeline = Compose(pipeline)

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])

        while True:
            data = self.prepare_train_img(self.img_ids[idx])
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)  # 在YTVOS类里进行了json.load
        self.cat_ids = self.ytvos.getCatIds()  # 40 categories
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]  # 得到json中videos字段中的一个元素，包括id height等
            info['filenames'] = info['file_names']
            # if '281629cb41/00055.jpg'in info['filenames']:
            #    import pdb
            #    pdb.set_trace()
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)  # 一系列ann组成的list
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):  # 实现了__len__方法
            vid_id, _ = self.img_ids[i]  # tuple (idx, frame_id))
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]

        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2 * np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2 * np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:, :2] + bbox[:, 2:]) / 2.
        sizes = bbox[:, 2:] - bbox[:, :2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes / 2.
        new_x2y2 = new_centers + new_sizes / 2.
        c_min = [0, 0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
            # check if the frame id is valid
            ref_idx = (vid, i)
            if i != frame_id and ref_idx in self.img_ids:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_img(self, idx):
        vid, frame_id = idx  # type(idx) == tuple
        vid_info = self.vid_infos[vid]
        _, ref_frame_id = self.sample_ref(idx)

        img_info = dict({'filename': vid_info['filenames'][frame_id],
                         'height': vid_info['height'],
                         'width': vid_info['width']
                         })
        ref_img_info = dict({'filename': vid_info['filenames'][ref_frame_id],
                             'height': vid_info['height'],
                             'width': vid_info['width']
                             })

        ann_info = self.get_ann_info(vid, frame_id)  # 得到了这一帧各个实例的ann
        ref_ann_info = self.get_ann_info(vid, ref_frame_id)

        results = dict(img_info=img_info, ann_info=ann_info, ids=ann_info['obj_ids'])
        ref_results = dict(img_info=ref_img_info, ann_info=ref_ann_info, ids=ref_ann_info['obj_ids'])
        self.pre_pipeline(results)
        self.pre_pipeline(ref_results)
        data = self.pipeline(results)
        ref_data = self.pipeline(ref_results)
        if data['gt_bboxes'].data.shape[0] == 0 or data['gt_labels'].data.shape[0] == 0:
            return None
        if ref_data['gt_bboxes'].data.shape[0] == 0 or ref_data['gt_labels'].data.shape[0] == 0:
            return None
        if data is None or ref_data is None:
            return None
        data['ref_data'] = ref_data
        
        gt_ids = data['img_metas'].data['ids']
        ref_ids = ref_data['img_metas'].data['ids']
        gt_pids = [ref_ids.index(i) + 1 if i in ref_ids else 0 for i in gt_ids]
        data['gt_pids'] = DC(torch.tensor(gt_pids))

        if len(gt_pids) == 0:
            return None

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx  # both start from 0
        vid_info = self.vid_infos[vid]

        img_info = dict(filename=vid_info[frame_id],
                        height=vid_info['height'],
                        weight=vid_info['weight']
                    )

        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        results['img_metas'].data['video_id'] = vid
        results['img_metas'].data['frame_id'] = frame_id
        results['img_metas'].data['is_first'] = (frame_id == 0)

        return data

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            # gt_mask_polys = []
            # gt_poly_lens = []

        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area

            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])  # ann的主键: id
                # self.cat2label: dict(cat_id: index) (index是从0开始的)
                # gt_labels记录属于哪一类
                gt_labels.append(self.cat2label[ann['category_id']])  # append的是index
                # TODO 检查这么写有没有问题，应该得保证gt_mask与gt_ids长度能对得上
                if with_mask:
                    gt_masks.append(segm)
                    # mask_polys = [
                    #     p for p in segm if len(p) >= 6
                    # ]  # valid polygons have >= 3 points (6 coordinates)
                    # poly_lens = [len(p) for p in mask_polys]
                    # gt_mask_polys.append(mask_polys)
                    # gt_poly_lens.extend(poly_lens)

        if gt_bboxes:  # gt_bboxes == []则为true
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)  # 变成二维数组了
            gt_labels = np.array(gt_labels, dtype=np.int64)
            try:
                test1 = gt_bboxes[0]
                test2 = gt_bboxes[0, 1]
            except:
                import pdb
                pdb.set_trace()
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks  # 三维数组
            # # poly format is not used in the current implementation
            # ann['mask_polys'] = gt_mask_polys  # 三维数组
            # ann['poly_lens'] = gt_poly_lens
        return ann
