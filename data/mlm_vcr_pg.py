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
from transformers import BertTokenizer

############################################################

def _pad_ids(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))


additional_special_tokens = ["person_0", "person_1", "person_2", "person_3", "person_4",
                                      "person_5", "person_6", "person_7", "person_8", "person_9",
                                      "person_10", "person_11", "person_12", "person_13", "person_14",
                                      "person_15", "person_16", "person_17", "person_18", "person_19",
                                      "person_20", "person_21", "person_22", "person_23", "person_24",
                                      "person_25", "person_26", "person_27", "person_28", "person_29",
                                      "person_30", "person_31", "person_32", "person_33", "person_34",
                                      "person_35", "person_36", "person_37", "person_38", "person_39",
                                      "person_40", "person_41", "person_42", "person_43", "person_44",
                                      "person_45", "person_46", "person_47", "person_48", "person_49",
                                      "person_50", "person_51", "person_52", "person_53", "person_54",
                                      "person_55", "person_56", "person_57", "person_58", "person_59",
                                      "person_60", "person_61", "person_62", "person_63", "person_64",
                                      "person_65", "person_66", "person_67", "person_68", "person_69",
                                      "person_70", "person_71", "person_72", "person_73", "person_74",
                                      "person_75", "person_76", "person_77", "person_78", "person_79", "person_80"]




def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))
            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class VcrTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120, task="qa,qar"):
        assert task == "q" or task == "qa" or task == "qar" or task == "qa,qar", \
            "VCR only support the following tasks: 'q', 'qa', 'qar' or 'qa,qar'"
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
                #if len_ <= max_txt_len
            }

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']  # CLS
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']


class VcrDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db_gt=None, img_db=None):
        # assert not (img_db_gt is None and img_db is None), \
        #     "img_db_gt and img_db cannot all be None"
        # assert isinstance(txt_db, VcrTxtTokLmdb)
        # assert img_db_gt is None or isinstance(img_db_gt, DetectFeatLmdb)
        # assert img_db is None or isinstance(img_db, DetectFeatLmdb)
        #
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

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)  # img_feat_gt = bbx, img-feat = whole img
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


class VcrMlmDataset(VcrDetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    # def __init__(self, txt_db, img_gt, img_db, neg_sample_p=0.5):
    def __init__(self, txt_db, img_db_gt=None, img_db=None, person_mask=True):
        super().__init__(txt_db, img_db_gt, img_db)
        assert self.task != "qa,qar", \
            "loading training dataset with each task separately"
        # self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.person_mask = person_mask
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
        special_tokens_dict = {
            'additional_special_tokens': additional_special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

    def _get_input_ids(self, txt_dump):
        # text input
        answer_label = txt_dump['qa_target']
        rationale_label = txt_dump['qar_target']

        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_masks = txt_dump['input_ids_as_masks']

        input_ids_rs = txt_dump['input_ids_rs']
        input_ids_rs_masks = txt_dump['input_ids_rs_masks']
        assert answer_label >= 0, "answer_label < 0"
        input_ids_q = [self.txt_db.cls_] + txt_dump['input_ids'] + [self.txt_db.sep] #+ copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
        input_ids_masks_q = [-1] + txt_dump['input_ids_masks'] + [-1] #+ copy.deepcopy(input_ids_as_masks[answer_label]) + [-1]
        type_ids_q = [0] * len(input_ids_q)

        if self.task == "qa":
            input_ids_gt_a = copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
            input_ids_gt_a_mask = copy.deepcopy(input_ids_as_masks[answer_label]) + [-1]
            type_ids_gt_a = [1] * len(input_ids_gt_a)

            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_masks_q += input_ids_gt_a_mask

        if self.task == "qar":
            input_ids_gt_r = copy.deepcopy(input_ids_rs[rationale_label]) + [self.txt_db.sep]
            input_ids_gt_r_mask = copy.deepcopy(input_ids_rs_masks[rationale_label]) + [-1]
            type_ids_gt_r = [2] * len(input_ids_gt_r)

            type_ids_q += type_ids_gt_r
            input_ids_q += input_ids_gt_r
            input_ids_masks_q += input_ids_gt_r_mask

        assert len(input_ids_q) == len(type_ids_q)
        if self.person_mask:
            input_ids_q, txt_labels = self.create_mlm_io(input_ids_q,
                                                         input_ids_masks_q)  # , person_negative_input_ids, person_swap_in_image)
        else:

            input_ids_q, txt_labels = self.create_random_mlm_io(input_ids_q)
        assert len(input_ids_q) == len(type_ids_q) == len(txt_labels)
        return input_ids_q, type_ids_q, txt_labels

    def create_random_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids[1:-1],
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        assert len(input_ids) == len(txt_labels)
        return input_ids, txt_labels

    def create_mlm_io(self, tokens, input_ids_masks_q):  # , person_negative_input_ids, person_swap_in_image):

        output_label = []
        mask = self.txt_db.mask
        vocab_range = self.txt_db.v_range
        person_mask_candidates = sum(torch.tensor(input_ids_masks_q) >= 0).float()
        mask_budget = round(len(tokens)*0.15)
        person_mask_budget = min(round(mask_budget) , person_mask_candidates)
        common_mask_budget = max(mask_budget - person_mask_budget, 1)

        person_mask_ratio = person_mask_budget / person_mask_candidates
        common_mask_ratio = common_mask_budget/(len(tokens)-person_mask_candidates)


        for i, token in enumerate(tokens):

            # if tokens[i] == self.txt_db.cls_ or tokens[i] == self.txt_db.sep:
            #     output_label.append(-1)
            #     continue

            if input_ids_masks_q[i] >= 0 :
                prob = random.random()
                if self.person_mask and prob < person_mask_ratio:  # and person_swap_in_image:

                    output_label.append(token)
                    # person_range = set(range(28996, 29077)) - set([token])
                    # mask = random.choice(list(person_range))
                    tokens[i] = mask

                else:
                    output_label.append(-1)
            #########
            # else:
            #     output_label.append(-1)
            ########
            elif input_ids_masks_q[i] < 0:
                prob = random.random()
                # mask token with 15% probability
                if prob < common_mask_ratio:
                    prob /= common_mask_ratio
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(list(range(*vocab_range)))
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    output_label.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)

        if all(o == -1 for o in output_label):
            # at least mask 1
            output_label[0] = tokens[0]
            tokens[0] = mask

        assert len(tokens) == len(output_label)
        return tokens, output_label

    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'][0], example['img_fname'][1])
        # text input
        input_ids_q, type_ids_q, txt_labels = self._get_input_ids(example)
        attn_masks = torch.ones(len(input_ids_q) + num_bb, dtype=torch.long)
        input_ids = torch.tensor(input_ids_q)
        txt_type_ids = torch.tensor(type_ids_q)
        txt_labels = torch.tensor(txt_labels)

        objects = example['objects']

        person_list = np.where(np.array(objects) == 'person')[0]
        person_ids = len(objects) * [0] + [0]  # add whole image
        for i, pid in enumerate(person_list):
            person_ids[i] = self.tokenizer.convert_tokens_to_ids(['person_%d' % pid])[0]

        person_ids = _pad_ids(person_ids, num_bb)
        person_ids = torch.LongTensor(person_ids)
        boxes_mask = [1] * num_bb
        boxes_mask = torch.LongTensor(boxes_mask)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_type_ids, txt_labels, person_ids, boxes_mask


def vcr_mlm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_type_ids, txt_labels, person_ids, boxes_mask
     ) = map(list, unzip(inputs))
    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    person_ids = pad_sequence(person_ids, batch_first=True, padding_value=0)
    boxes_mask = pad_sequence(boxes_mask, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()  # max_tl 은 패딩 포함, out_size 는 패딩 삭제
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels,
             'person_ids': person_ids,
             'boxes_mask': boxes_mask
             }
    return batch


