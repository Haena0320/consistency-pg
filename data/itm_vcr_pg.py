"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
from collections import defaultdict
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np
from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb, TxtLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .sampler import TokenBucketSampler
import json
import numpy as np


class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class VcrTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120, task="qa,qar"):
        assert task == "qa" or task == "qar" or task == "qa,qar", \
            "VCR only support the following tasks: 'qa', 'qar' or 'qa,qar'"
        self.task = task
        if task == "qa,qar":
            id2len_task = "qar"
        else:
            id2len_task = task
        if max_txt_len == -1:
            self.id2len = json.load(
                open(f'{db_dir}/id2len_{id2len_task}.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(
                    open(f'{db_dir}/id2len_{id2len_task}.json')
                ).items()
                if len_ <= max_txt_len
            }

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']


class VcrDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db_gt=None, img_db=None):
        assert not (img_db_gt is None and img_db is None), \
            "img_db_gt and img_db cannot all be None"
        # assert isinstance(txt_db, VcrTxtTokLmdb)
        # assert img_db_gt is None or isinstance(img_db_gt, DetectFeatLmdb)
        # assert img_db is None or isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.img_db_gt = img_db_gt
        self.task = self.txt_db.task
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img

        if self.img_db and self.img_db_gt:
            self.lens = [tl + self.img_db_gt.name2nbb[txt2img[id_][0]] +
                         self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db:
            self.lens = [tl + self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        else:
            self.lens = [tl + self.img_db_gt.name2nbb[txt2img[id_][0]]
                         for tl, id_ in zip(txt_lens, self.ids)]

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            img_feat_gt, bb_gt = self.img_db_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5] * bb_gt[:, 5:]], dim=-1)

            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_db:
            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)

        elif self.img_db_gt:
            img_feat, bb = self.img_db_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb


class VcrItmDataset(VcrDetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    # def __init__(self, txt_db, img_gt, img_db, neg_sample_p=0.5):
    def __init__(self, txt_db, img_db_gt=None, img_db=None, person_corrupt=True, neg_sample_p=0.5):
        super().__init__(txt_db, img_db_gt, img_db)
        assert self.task != "qa,qar", \
            "loading training dataset with each task separately"
        # self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.neg_sample_p = neg_sample_p
        self.corrupt_method = "person_corrupt" if person_corrupt else "random_corrupt"
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1 - self.neg_sample_p])
        self.swap_labels = [0] * len(self.labels)
        # self.lens = []
        # self.train_imgs = []
        # for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
        #     img_fname = super().__getitem__(i)['img_fname']
        #     #if self.labels[i] == 0: # image
        #         #img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
        #         ### question text [person1]-> [person2]
        #
        #     self.train_imgs.append(img_fname)
        #     self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = [self.txt_db.cls_] + txt_dump['input_ids'] + [self.txt_db.sep]
        type_ids_q = [0] * len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as

        assert len(input_ids_q) == len(type_ids_q)
        return input_ids_q, input_ids_for_choices, type_ids_q

    def sample_negative(self, txt_dump, idx, ng_method):
        i = copy.deepcopy(idx)

        if ng_method == "random_corrupt" or txt_dump['person_negative'] < 1:
            # if flip < 0.3 or txt_dump['person_negative'] < 1:
            while i == idx:
                i = np.random.randint(len(self.ids))
            example_negative = super().__getitem__(i)
            return self._get_input_ids(example_negative)

        # elif flip >= 0.3:
        elif ng_method == "person_corrupt":
            return self._person_negative_input_ids(txt_dump, idx)

        else:
            raise NotImplementedError
        ###
        # if txt_dump["person_negative"] == 0:
        #     while i == idx:
        #         i = np.random.randint(len(self.ids))
        #     example_negative = super().__getitem__(i)
        #     return self._get_input_ids(example_negative)
        # else: # swap person
        #     return self._person_negative_input_ids(txt_dump, idx)

    def _person_negative_input_ids(self, txt_dump, idx):
        # swap label == 1
        self.swap_labels[idx] = 1

        # text input
        input_ids_q = [self.txt_db.cls_] + txt_dump['person_negative_input_ids'] + [self.txt_db.sep]
        type_ids_q = [0] * len(input_ids_q)

        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = copy.deepcopy(
                txt_dump['person_negative_input_ids_gt']) + [self.txt_db.sep]
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as
        assert len(input_ids_q) == len(type_ids_q)

        return input_ids_q, input_ids_for_choices, type_ids_q

    def _get_person_masks(self, input_ids_masks, num_bb):
        '''
        :param input_ids_masks: [-1, -1, 0, -1, -1, 2, -1, -1] # [1, text_len]
        :param num_bb: 5
        :return: person_masks: [[0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0]], # [num_person_tag, text_len]
        :return: object_mask : [[1,0,0,0,0], [0,0,1,0,0]]  #[num_person_tag, num_bb]
        '''

        input_ids_masks = np.array(input_ids_masks)

        person_idx = np.where(input_ids_masks >= 0)[0]
        person_masks = np.zeros([len(person_idx), len(input_ids_masks)])
        person_masks[np.arange(len(person_idx)), person_idx] = 1

        person_tag = input_ids_masks[input_ids_masks >= 0]
        object_masks = np.zeros([len(person_tag), num_bb])
        object_masks[np.arange(len(person_tag)), person_tag] = 1

        return person_masks.tolist(), object_masks.tolist()

    def __getitem__(self, i):
        """
         [[txt, img1],
          [txt, img2]]
         """
        qid = self.ids[i]
        example = super().__getitem__(i)

        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]

        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        # text input
        origin_input_ids_q, origin_input_ids_for_choices, origin_type_ids_q = self._get_input_ids(
            example)  ## train set ##
        corrupt_input_ids_q, corrupt_input_ids_for_choices, corrupt_type_ids_q = self.sample_negative(example, i,
                                                                                                      self.corrupt_method)

        # person_masks, object_masks = self._get_person_masks(example['input_ids_masks'], num_bb)

        if self.labels[i] == 0:
            input_ids_q = corrupt_input_ids_q
            input_ids_for_choices = corrupt_input_ids_for_choices
            type_ids_q = corrupt_type_ids_q
        elif self.labels[i] == 1:
            input_ids_q = origin_input_ids_q
            input_ids_for_choices = origin_input_ids_for_choices
            type_ids_q = origin_type_ids_q
        else:
            raise Exception

        ground_truth_swap_label = self.swap_labels[i]
        attn_masks = torch.ones(len(input_ids_q) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        swap_label = torch.Tensor(1).long()
        swap_label.data.fill_(ground_truth_swap_label)
        input_ids = torch.tensor(input_ids_q)
        origin_input_ids_q = torch.tensor(origin_input_ids_q)
        txt_type_ids = torch.tensor(type_ids_q)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_type_ids, swap_label, target, qid, origin_input_ids_q


def vcr_itm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_type_ids, swap_labels, targets, qids, origin_input_ids_q
     ) = map(list, unzip(inputs))
    txt_lens = [i.size(0) for i in input_ids]
    origin_input_ids_qs = pad_sequence(origin_input_ids_q, batch_first=True, padding_value=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    swap_labels = torch.cat(swap_labels, dim=0)
    targets = torch.cat(targets, dim=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'swap_labels': swap_labels,  # ㅍㅛ기용
             'origin_input_ids_qs': origin_input_ids_qs,
             'targets': targets}
    return batch


class VcrItmValDataset(VcrDetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    # def __init__(self, txt_db, img_gt, img_db, neg_sample_p=0.5):
    def __init__(self, txt_db, img_db_gt=None, img_db=None, person_corrupt=True, neg_sample_p=0.5):
        super().__init__(txt_db, img_db_gt, img_db)
        assert self.task != "qa,qar", \
            "loading training dataset with each task separately"
        # self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.neg_sample_p = neg_sample_p
        self.corrupt_method = "person_corrupt" if person_corrupt else "random_corrupt"
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1 - self.neg_sample_p])
        self.swap_labels = [0] * len(self.labels)
        # self.lens = []
        # self.train_imgs = []
        # for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
        #     img_fname = super().__getitem__(i)['img_fname']
        #     #if self.labels[i] == 0: # image
        #         #img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
        #         ### question text [person1]-> [person2]
        #
        #     self.train_imgs.append(img_fname)
        #     self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = [self.txt_db.cls_] + txt_dump['input_ids'] + [self.txt_db.sep]
        type_ids_q = [0] * len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']

        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
            input_ids_q += input_ids_gt_a

            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a

            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as

        assert len(input_ids_q) == len(type_ids_q)
        return input_ids_q, input_ids_for_choices, type_ids_q

    def sample_negative(self, txt_dump, idx, ng_method):
        i = copy.deepcopy(idx)

        #####
        if ng_method == "random_corrupt" or txt_dump['person_negative'] < 1:
            while i == idx:
                i = np.random.randint(len(self.ids))
            example_negative = super().__getitem__(i)
            self.swap_labels[idx] = -1
            return self._get_input_ids(example_negative)

        elif ng_method == "person_corrupt":
            return self._person_negative_input_ids(txt_dump, idx)

        else:
            raise NotImplementedError
        ###
        # if txt_dump["person_negative"] == 0:
        #     while i == idx:
        #         i = np.random.randint(len(self.ids))
        #     example_negative = super().__getitem__(i)
        #     return self._get_input_ids(example_negative)
        # else: # swap person
        #     return self._person_negative_input_ids(txt_dump, idx)

    def _person_negative_input_ids(self, txt_dump, idx):
        # swap label == 1
        self.swap_labels[idx] = 1
        # text input
        input_ids_neg_q = [self.txt_db.cls_] + copy.deepcopy(txt_dump['person_negative_input_ids']) + [self.txt_db.sep]
        type_ids_neg_q = [0] * len(input_ids_neg_q)
        assert len(txt_dump['person_negative_input_ids_gt']) != 0

        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_neg_gt_a = copy.deepcopy(txt_dump['person_negative_input_ids_gt']) + [self.txt_db.sep]
            input_ids_neg_q += input_ids_neg_gt_a

            type_ids_neg_gt_a = [2] * len(input_ids_neg_gt_a)
            type_ids_neg_q += type_ids_neg_gt_a

            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as
        assert len(input_ids_neg_q) == len(type_ids_neg_q)

        return input_ids_neg_q, input_ids_for_choices, type_ids_neg_q

    def _get_person_masks(self, input_ids_masks, num_bb):
        '''
        :param input_ids_masks: [-1, -1, 0, -1, -1, 2, -1, -1] # [1, text_len]
        :param num_bb: 5
        :return: person_masks: [[0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0]], # [num_person_tag, text_len]
        :return: object_mask : [[1,0,0,0,0], [0,0,1,0,0]]  #[num_person_tag, num_bb]
        '''

        input_ids_masks = np.array(input_ids_masks)

        person_idx = np.where(input_ids_masks >= 0)[0]
        person_masks = np.zeros([len(person_idx), len(input_ids_masks)])
        person_masks[np.arange(len(person_idx)), person_idx] = 1

        person_tag = input_ids_masks[input_ids_masks >= 0]
        object_masks = np.zeros([len(person_tag), num_bb])
        object_masks[np.arange(len(person_tag)), person_tag] = 1

        return person_masks.tolist(), object_masks.tolist()

    def __getitem__(self, i):
        """
         [[txt, img1],
          [txt, img2]]
         """
        qid = self.ids[i]
        example = super().__getitem__(i)

        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]

        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        # text input
        origin_input_ids_q, origin_input_ids_for_choices, origin_type_ids_q = self._get_input_ids(
            example)  ### valid set ##
        corrupt_input_ids_q, corrupt_input_ids_for_choices, corrupt_type_ids_q = self.sample_negative(example, i,
                                                                                                      self.corrupt_method)

        # person_masks, object_masks = self._get_person_masks(example['input_ids_masks'], num_bb)

        # if self.labels[i] == 0:
        #     input_ids_q = corrupt_input_ids_q
        #     input_ids_for_choices = corrupt_input_ids_for_choices
        #     type_ids_q = corrupt_type_ids_q
        # elif self.labels[i] == 1:
        #     input_ids_q = origin_input_ids_q
        #     input_ids_for_choices = origin_input_ids_for_choices
        #     type_ids_q = origin_type_ids_q
        # else:
        #     raise Exception

        ground_truth_swap_label = self.swap_labels[i]
        origin_attn_masks = torch.ones(len(origin_input_ids_q) + num_bb, dtype=torch.long)
        corrupt_attn_masks = torch.ones(len(corrupt_input_ids_q) + num_bb, dtype=torch.long)

        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)
        swap_label = torch.Tensor(1).long()
        swap_label.data.fill_(ground_truth_swap_label)

        origin_input_ids_q = torch.tensor(origin_input_ids_q)
        corrupt_input_ids_q = torch.tensor(corrupt_input_ids_q)

        origin_txt_type_ids = torch.tensor(origin_type_ids_q)
        corrupt_txt_type_ids = torch.tensor(corrupt_type_ids_q)

        return img_feat, img_pos_feat, origin_input_ids_q, corrupt_input_ids_q, origin_attn_masks, corrupt_attn_masks, origin_txt_type_ids, corrupt_txt_type_ids, swap_label  # , target, qid,


def vcr_itm_val_collate(inputs):
    (img_feats, img_pos_feats, origin_input_ids, corrupt_input_ids, origin_attn_masks, corrupt_attn_masks,
     origin_txt_type_ids, corrupt_txt_type_ids, swap_labels  # , target, qid,
     ) = map(list, unzip(inputs))

    origin_txt_lens = [i.size(0) for i in origin_input_ids]
    corrupt_txt_lens = [i.size(0) for i in corrupt_input_ids]

    origin_input_ids = pad_sequence(origin_input_ids, batch_first=True, padding_value=0)
    corrupt_input_ids = pad_sequence(corrupt_input_ids, batch_first=True, padding_value=0)
    origin_txt_type_ids = pad_sequence(origin_txt_type_ids, batch_first=True, padding_value=0)
    corrupt_txt_type_ids = pad_sequence(corrupt_txt_type_ids, batch_first=True, padding_value=0)

    origin_position_ids = torch.arange(0, origin_input_ids.size(1), dtype=torch.long).unsqueeze(0)
    corrupt_position_ids = torch.arange(0, corrupt_input_ids.size(1), dtype=torch.long).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    origin_attn_masks = pad_sequence(origin_attn_masks, batch_first=True, padding_value=0)
    corrupt_attn_masks = pad_sequence(corrupt_attn_masks, batch_first=True, padding_value=0)

    swap_labels = torch.cat(swap_labels, dim=0)
    # targets = torch.cat(targets, dim=0)

    bs, origin_max_tl = origin_input_ids.size()
    bs, corrupt_max_tl = corrupt_input_ids.size()

    origin_out_size = origin_attn_masks.size(1)
    corrupt_out_size = corrupt_attn_masks.size(1)
    origin_gather_index = get_gather_index(origin_txt_lens, num_bbs, bs, origin_max_tl, origin_out_size)
    corrupt_gather_index = get_gather_index(corrupt_txt_lens, num_bbs, bs, corrupt_max_tl, corrupt_out_size)

    batch = {'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,

             'origin_input_ids': origin_input_ids,
             'origin_txt_type_ids': origin_txt_type_ids,
             'origin_position_ids': origin_position_ids,
             'origin_attn_masks': origin_attn_masks,
             'origin_gather_index': origin_gather_index,

             'corrupt_input_ids': corrupt_input_ids,
             'corrupt_txt_type_ids': corrupt_txt_type_ids,
             'corrupt_position_ids': corrupt_position_ids,
             'corrupt_attn_masks': corrupt_attn_masks,
             'corrupt_gather_index': corrupt_gather_index,

             'swap_labels': swap_labels,  # person_corrupt, random_corrupt 표기용
             }
    return batch


############################################################


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len - tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad
