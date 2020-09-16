"""
data_json has 
0. refs        : list of {ref_id, ann_id, box, image_id, split, category_id, sent_ids}
1. images      : list of {image_id, ref_ids, ann_ids, file_name, width, height, h5_id}
2. anns        : list of {ann_id, category_id, image_id, box, h5_id}
3. sentences   : list of {sent_id, tokens, h5_id}
4: word_to_ix  : word->ix
5: cat_to_ix   : cat->ix
6: label_length: L
Note, box in [xywh] format
data_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py
import json
import random
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import os

from .coco import COCO

class Loader(COCO):

    def __init__(self, opt, split='train'):
        self.opt = opt
        self.data_path = opt.data_path
        self.coco_dir = os.path.join(self.data_path, "coco")
        self.img_dir = os.path.join(self.coco_dir, "images", "train2014")
        self.coco_json = opt.coco_json
        self.coco = coco.COCO(self.coco_json)
        # load the json file which contains info about the dataset
        print('Loader loading data.json: ', opt.data_json)
        self.info = json.load(open(opt.data_json))
        self.split = split
        self.word_to_ix = self.info['word_to_ix']
        self.vocab_size = len(self.word_to_ix)
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        print('vocab size is ', self.vocab_size)
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object cateogry size is ', len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        self.split_ref()
        print('we have %s images.' % len(self.images))
        print('we have %s anns.' % len(self.anns))
        print('we have %s refs.' % len(self.refs))
        print('we have %s sentences.' % len(self.sentences))
        print('label_length is ', self.label_length)

        # construct mapping
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs for sent_id in ref['sent_ids']}

        # read data_h5 if exists
        self.data_h5 = None
        if opt.data_h5 is not None:
            print('Loader loading data.h5: ', opt.data_h5)
            self.data_h5 = h5py.File(opt.data_h5, 'r')
            assert self.data_h5['labels'].shape[0] == len(self.sentences), 'label.shape[0] not match sentences'
            assert self.data_h5['labels'].shape[1] == self.label_length, 'label.shape[1] not match label_length'
    
    def split_ref(self):
        new_ref = []
        new_sent = []
        for r in self.refs:
            if r['split'] == self.split:
                new_ref.append(r)
                for s in r['sent_ids']:
                    new_sent.append(s)
        self.refs = new_ref
        self.sent_ids_split = new_sent
    
    def __len__(self):
        return len(self.sent_ids_split)


    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    @property
    def vocab_size(self):
        return len(self.word_to_ix)

    @property
    def label_length(self):
        return self.info['label_length']

    def encode_labels(self, sent_str_list):
        """Input:
        sent_str_list: list of n sents in string format
        return int32 (n, label_length) zeros padded in end
        """
        num_sents = len(sent_str_list)
        L = np.zeros((num_sents, self.label_length), dtype=np.int32)
        for i, sent_str in enumerate(sent_str_list):
            tokens = sent_str.split()
            for j, w in enumerate(tokens):
              if j < self.label_length:
                  L[i, j] = self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>']
        return L

    def decode_labels(self, labels):
        """
        labels: int32 (n, label_length) zeros padded in end
        return: list of sents in string format
        """
        decoded_sent_strs = []
        num_sents = labels.shape[0]
        for i in range(num_sents):
            label = labels[i].tolist()
            sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0])
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs

    def fetch_label(self, ref_id, num_sents):
        """
        return: int32 (num_sents, label_length) and picked_sent_ids
        """
        ref = self.Refs[ref_id]
        sent_ids = list(ref['sent_ids'])  # copy in case the raw list is changed
        seq = []
        if len(sent_ids) < num_sents:
            append_sent_ids = [random.choice(sent_ids) for _ in range(num_sents - len(sent_ids))]
            sent_ids += append_sent_ids
        else:
            sent_ids = sent_ids[:num_sents]
        assert len(sent_ids) == num_sents
        # fetch label
        for sent_id in sent_ids:
            sent_h5_id = self.Sentences[sent_id]['h5_id']
            seq += [self.data_h5['labels'][sent_h5_id, :]]
        seq = np.vstack(seq)
        return seq, sent_ids

    def fetch_seq(self, sent_id):
        # return int32 (label_length, )
        sent_h5_id = self.Sentences[sent_id]['h5_id']
        seq = self.data_h5['labels'][sent_h5_id, :]
        return seq

#class COCO(data.Dataset):
#  num_classes = 80
#  default_resolution = [512, 512]
#  mean = np.array([0.40789654, 0.44719302, 0.47026115],
#                   dtype=np.float32).reshape(1, 1, 3)
#  std  = np.array([0.28863828, 0.27408164, 0.27809835],
#                   dtype=np.float32).reshape(1, 1, 3)
#
#  def __init__(self, opt, split):
#    super(COCO, self).__init__()
#    self.data_dir = os.path.join(opt.data_dir, 'coco')
#    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
#    if split == 'test':
#      self.annot_path = os.path.join(
#          self.data_dir, 'annotations', 
#          'image_info_test-dev2017.json').format(split)
#    else:
#      if opt.task == 'exdet':
#        self.annot_path = os.path.join(
#          self.data_dir, 'annotations', 
#          'instances_extreme_{}2017.json').format(split)
#      else:
#        self.annot_path = os.path.join(
#          self.data_dir, 'annotations', 
#          'instances_{}2017.json').format(split)
#    self.max_objs = 128
#    self.class_name = [
#      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
#      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#      'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#    self._valid_ids = [
#      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
#      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
#      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
#      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
#      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
#      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
#      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
#      82, 84, 85, 86, 87, 88, 89, 90]
#    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
#    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
#                      for v in range(1, self.num_classes + 1)]
#    self._data_rng = np.random.RandomState(123)
#    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
#                             dtype=np.float32)
#    self._eig_vec = np.array([
#        [-0.58752847, -0.69563484, 0.41340352],
#        [-0.5832747, 0.00994535, -0.81221408],
#        [-0.56089297, 0.71832671, 0.41158938]
#    ], dtype=np.float32)
#    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
#    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
#
#    self.split = split
#    self.opt = opt
#
#    print('==> initializing coco 2017 {} data.'.format(split))
#    self.coco = coco.COCO(self.annot_path)
#    self.images = self.coco.getImgIds()
#    self.num_samples = len(self.images)
#
#    print('Loaded {} {} samples'.format(split, self.num_samples))
#
#  def _to_float(self, x):
#    return float("{:.2f}".format(x))
#
#  def convert_eval_format(self, all_bboxes):
#    # import pdb; pdb.set_trace()
#    detections = []
#    for image_id in all_bboxes:
#      for cls_ind in all_bboxes[image_id]:
#        category_id = self._valid_ids[cls_ind - 1]
#        for bbox in all_bboxes[image_id][cls_ind]:
#          bbox[2] -= bbox[0]
#          bbox[3] -= bbox[1]
#          score = bbox[4]
#          bbox_out  = list(map(self._to_float, bbox[0:4]))
#
#          detection = {
#              "image_id": int(image_id),
#              "category_id": int(category_id),
#              "bbox": bbox_out,
#              "score": float("{:.2f}".format(score))
#          }
#          if len(bbox) > 5:
#              extreme_points = list(map(self._to_float, bbox[5:13]))
#              detection["extreme_points"] = extreme_points
#          detections.append(detection)
#    return detections
#
#  def __len__(self):
#    return self.num_samples
#
#  def save_results(self, results, save_dir):
#    json.dump(self.convert_eval_format(results), 
#                open('{}/results.json'.format(save_dir), 'w'))
#  
#  def run_eval(self, results, save_dir):
#    # result_json = os.path.join(save_dir, "results.json")
#    # detections  = self.convert_eval_format(results)
#    # json.dump(detections, open(result_json, "w"))
#    self.save_results(results, save_dir)
#    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
#    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
#    coco_eval.evaluate()
#    coco_eval.accumulate()
#    coco_eval.summarize()
