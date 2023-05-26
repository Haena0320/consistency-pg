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
from transformers import BertTokenizer
import json
import numpy as np
from copy import deepcopy
from cytoolz import curry
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

def _pad_ids(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))

@curry
def bert_tokenize(tokenizer, text_list, name, not_in_target_image=False):
    new_text = []
    person_masks = []
    for word in text_list:
        if isinstance(word, list):
            for e in word:
                if not_in_target_image:
                    tmp = f"person_{e}"
                    person_masks.append(int(e))
                else:
                    if name[int(e)] == "person":
                        tmp = name[int(e)] + f'_{e}'
                        person_masks.append(int(e))
                    else:
                        tmp = name[int(e)]
                        person_masks.append(-1)
                new_text.append(tmp)

        else:
            tokens = tokenizer.tokenize(word)
            new_text.extend(tokens)
            person_masks.extend([-1]*len(tokens))
    ids = tokenizer.convert_tokens_to_ids(new_text)
    assert len(ids) == len(person_masks)
    return ids, person_masks

toker = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
special_tokens_dict = {
    'additional_special_tokens': additional_special_tokens}
num_added_toks = toker.add_special_tokens(special_tokens_dict)

tokenizer = bert_tokenize(toker)

person_candidates = list(range(len(additional_special_tokens)))

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
        assert task == "q" or task == "qa" or task == "qar" or task == "qa,qar", \
            "VCR only support the following tasks: 'qa', 'qar' or 'qa,qar'"
        self.task = task
        if task == "qa,qar":
            id2len_task = "qar"
        else:
            id2len_task = "qa" if task in ["q", "qa"] else task #task

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
        self.cls_ = meta['ITM_CLS']  # CLS
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
        #print(txt2img)
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

def change_person_priors_in_choice(choice, tok_c, tok_orig):
    new_answer = []
    for tok_ in choice:
        if isinstance(tok_, list):
            if tok_ == tok_orig:
                tok__ = tok_c
            elif tok_ == tok_c:

                tok__ = deepcopy(tok_orig)
            else:
                tok__ = tok_
            new_answer.append(tok__)
        else:
            new_answer.append(tok_)
    return new_answer

def change_person_prior(data_one, person_candidates):
    input_all = data_one["objects"]
    rest_list = sorted(list(filter(lambda x: input_all[x] == "person", range(len(input_all)))))  # person index 만

    new_question = []
    new_answer = []
    new_rationale = []
    cnt = 0
    cng_tok = None
    not_in_target_image = False
    for tok in data_one['question']:
        if isinstance(tok, list):
            cnt += 1
            if cnt == 1:
                tok_orig = deepcopy(tok)
                candidates = list(set(rest_list) - set(tok))
                if len(candidates) > 0:
                    # print(candidates)
                    tok = [random.choice(candidates)]
                    if len(tok_orig) > 1:
                        tok.append(random.choice(tok_orig))
                    cng_tok = deepcopy(tok)
                    new_answer = change_person_priors_in_choice(data_one["answer_gt"], tok, tok_orig)
                    new_rationale = change_person_priors_in_choice(data_one["rationale_gt"], tok, tok_orig)
                elif len(candidates) == 0:
                    #print("no person last in image, let's choose one not in image ")
                    not_in_target_image = True
                    person_candidates = list(set(person_candidates) - set(tok))
                    tok = random.sample(person_candidates, len(tok))
                    assert tok != tok_orig
                    new_answer = change_person_priors_in_choice(data_one["answer_gt"], tok, tok_orig)
                    new_rationale = change_person_priors_in_choice(data_one["rationale_gt"], tok,
                                                                   tok_orig)
            else:
                if cng_tok == tok:
                    tok = deepcopy(tok_orig)

        new_question.append(tok)

    assert len(new_answer) != 0
    assert len(new_question) != 0
    return new_question, new_answer, new_rationale, not_in_target_image


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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
        special_tokens_dict = {
        'additional_special_tokens': additional_special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice([0, 1], size=len(self.ids), p=[self.neg_sample_p, 1 - self.neg_sample_p])
        ############################################################################################################

        # self.labels = np.random.choice(
        #     [0, 1, 2], size=len(self.ids), p=[0.38, 0.38, 0.24])
        #     #p=[self.neg_sample_p, 1 - self.neg_sample_p])
        # self.trick_labels = len(self.labels) * [0]

        ############################################################################################################
        # # import torch
        # first_round_id_q_ls = torch.load("./first_round_id_q.pkl")
        # if len(self.ids) != 212923:
        #
        #     self.labels = np.random.choice(
        #         [0, 1], size=len(self.ids),
        #         p=[self.neg_sample_p, 1 - self.neg_sample_p])
        #
        # # torch.save(first_round_id_q_ls, "first_round_id_q.pkl", _use_new_zipfile_serialization=False)
        # else:
        #     self.labels = [0] * len(self.ids)
        #     for id_ in first_round_id_q_ls:
        #         self.labels[id_] = 1

        ############################################################################################################


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

        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        answer_label = txt_dump['qa_target']
        rationale_label = txt_dump['qar_target']

        input_ids_q = [self.txt_db.cls_] + txt_dump['input_ids'] + [self.txt_db.sep] #+ copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
        type_ids_q = [0] * len(input_ids_q)

        if self.task =="qa":
            input_ids_gt_a = copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
            type_ids_gt_a = [1] * len(input_ids_gt_a)

            input_ids_q += input_ids_gt_a
            type_ids_q += type_ids_gt_a

        if self.task == "qar":
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_r = copy.deepcopy(input_ids_rs[rationale_label]) + [self.txt_db.sep]
            type_ids_gt_r = [2] * len(input_ids_gt_r)

            type_ids_q += type_ids_gt_r
            input_ids_q += input_ids_gt_r

        assert len(input_ids_q) == len(type_ids_q)
        return input_ids_q, type_ids_q

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

    def _person_negative_input_ids(self, txt_dump, idx):
        # swap label == 1
        self.swap_labels[idx] = 1
        # input_ids_as = txt_dump['input_ids_as']
        # input_ids_rs = txt_dump['input_ids_rs']
        names = txt_dump['objects']
        answer_label = txt_dump['qa_target']

        swapped_question, swapped_answer, swapped_rationale, not_in_target_image = change_person_prior(txt_dump,person_candidates)
        person_negative_input_ids, person_negative_input_ids_masks = tokenizer(swapped_question, names,
                                                                               not_in_target_image)
        person_negative_input_ids_gt, person_negative_input_ids_gt_masks = tokenizer(swapped_answer, names,
                                                                                     not_in_target_image)
        person_negative_input_ids_rs_gt, person_negative_input_ids_rs_gt_masks = tokenizer(swapped_rationale, names,
                                                                                           not_in_target_image)

        # text input
        input_ids_q = [self.txt_db.cls_] + person_negative_input_ids + [self.txt_db.sep]
        type_ids_q = [0] * len(input_ids_q)

        if self.task == "qa":
            input_ids_gt_a = person_negative_input_ids_gt + [self.txt_db.sep]
            type_ids_gt_a = [1]*len(input_ids_gt_a)
            input_ids_q += input_ids_gt_a
            type_ids_q += type_ids_gt_a

        if self.task == "qar":
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_r = person_negative_input_ids_rs_gt + [self.txt_db.sep]
            type_ids_gt_r = [2] * len(input_ids_gt_r)
            type_ids_q += type_ids_gt_r
            input_ids_q += input_ids_gt_r

        assert len(input_ids_q) == len(type_ids_q)
        return input_ids_q, type_ids_q

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
        origin_input_ids_q, origin_type_ids_q = self._get_input_ids(example)  ## train set ##
        corrupt_input_ids_q, corrupt_type_ids_q = self.sample_negative(example, i, self.corrupt_method)

        #person_masks, object_masks = self._get_person_masks(example['input_ids_masks'], num_bb)
        if self.labels[i] == 0:

            input_ids_q = corrupt_input_ids_q
            # input_ids_for_choices = corrupt_input_ids_for_choices
            type_ids_q = corrupt_type_ids_q

        elif self.labels[i] == 1:
            input_ids_q = origin_input_ids_q
            # input_ids_for_choices = origin_input_ids_for_choices
            type_ids_q = origin_type_ids_q

        else:
            # trick sample
            a = random.random()
            if a > 0.5: # 0 -> 2
                self.trick_labels[i] = 1
                input_ids_q = origin_input_ids_q
                type_ids_q = origin_type_ids_q

            else: # 1 -> 2
                self.trick_labels[i] = 1
                input_ids_q = corrupt_input_ids_q
                type_ids_q = corrupt_type_ids_q

        ground_truth_swap_label = self.swap_labels[i]
        attn_masks = torch.ones(len(input_ids_q) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        swap_label = torch.Tensor(1).long()
        swap_label.data.fill_(ground_truth_swap_label)
        input_ids = torch.tensor(input_ids_q)
        origin_input_ids_q = torch.tensor(origin_input_ids_q)
        txt_type_ids = torch.tensor(type_ids_q)

        objects = example['objects']

        person_list = np.where(np.array(objects) == 'person')[0]
        person_ids = len(objects) * [0] + [0]  # add whole image
        for i, pid in enumerate(person_list):
            person_ids[i] = self.tokenizer.convert_tokens_to_ids(['person_%d' % pid])[0]

        person_ids = _pad_ids(person_ids, num_bb)
        person_ids = torch.LongTensor(person_ids)
        boxes_mask = [1] * num_bb
        boxes_mask = torch.LongTensor(boxes_mask)

        # trick_label = torch.Tensor(1).long"""
        # Copyright (c) Microsoft Corporation.
        # Licensed under the MIT license.
        #


        return input_ids, img_feat, img_pos_feat, attn_masks, txt_type_ids, swap_label, target, qid, origin_input_ids_q, person_ids, boxes_mask#, trick_label


def vcr_itm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_type_ids,
     swap_labels, targets, qids, origin_input_ids_q,
     person_ids, boxes_mask #, trick_label
     ) = map(list, unzip(inputs))
    txt_lens = [i.size(0) for i in input_ids]
    origin_input_ids_qs = pad_sequence(origin_input_ids_q, batch_first=True, padding_value=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    person_ids = pad_sequence(person_ids, batch_first=True, padding_value=0)
    boxes_mask = pad_sequence(boxes_mask, batch_first=True, padding_value=0)

    swap_labels = torch.cat(swap_labels, dim=0)
    targets = torch.cat(targets, dim=0)
    #trick_labels = torch.cat(trick_label, dim=0)

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
             'targets': targets,
             'person_ids': person_ids,
             'boxes_mask': boxes_mask
             ,
             #'trick_labels': trick_labels
            }
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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
        special_tokens_dict = {
            'additional_special_tokens': additional_special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

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
        answer_label = txt_dump['qa_target']

        if self.task =="qa":
            input_ids_gt_a = copy.deepcopy(input_ids_as[answer_label]) + [self.txt_db.sep]
            type_ids_gt_a = [1] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a

        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            input_ids_gt_r = copy.deepcopy(input_ids_rs[txt_dump['qar_target']]) + [self.txt_db.sep]
            type_ids_gt_r = [2] * len(input_ids_gt_r)
            type_ids_q += type_ids_gt_r
            input_ids_q += input_ids_gt_r

            assert answer_label >= 0, "answer_label < 0"
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
        names = txt_dump['objects']
        answer_label = txt_dump['qa_target']

        swapped_question, swapped_answer, swapped_rationale, not_in_target_image = change_person_prior(txt_dump,person_candidates)
        person_negative_input_ids, person_negative_input_ids_masks = tokenizer(swapped_question, names,
                                                                               not_in_target_image)
        person_negative_input_ids_gt, person_negative_input_ids_gt_masks = tokenizer(swapped_answer, names,
                                                                                     not_in_target_image)
        person_negative_input_ids_rs_gt, person_negative_input_ids_rs_gt_masks = tokenizer(swapped_rationale, names,
                                                                                           not_in_target_image)





        input_ids_neg_q = [self.txt_db.cls_] + person_negative_input_ids + [self.txt_db.sep]
        type_ids_neg_q = [0] * len(input_ids_neg_q)
        assert len(input_ids_neg_q) == len(type_ids_neg_q)
        assert len(person_negative_input_ids_gt) != 0

        input_ids_as = txt_dump['input_ids_as']

        if self.task =="qa":
            input_ids_neg_gt_a = person_negative_input_ids_gt + [self.txt_db.sep]
            type_ids_neg_gt_a = [1] * len(input_ids_neg_gt_a)

            input_ids_neg_q += input_ids_neg_gt_a
            type_ids_neg_q += type_ids_neg_gt_a

            assert len(input_ids_neg_q) == len(type_ids_neg_q)

        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']

            #assert answer_label >= 0, "answer_label < 0"
            input_ids_neg_gt_r = person_negative_input_ids_rs_gt + [self.txt_db.sep]
            type_ids_neg_gt_r = [2] * len(input_ids_neg_gt_r)

            input_ids_neg_q += input_ids_neg_gt_r
            type_ids_neg_q += type_ids_neg_gt_r

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

        objects = example['objects']

        person_list = np.where(np.array(objects) == 'person')[0]
        person_ids = len(objects) * [0] + [0]  # add whole image
        for i, pid in enumerate(person_list):
            person_ids[i] = self.tokenizer.convert_tokens_to_ids(['person_%d' % pid])[0]

        person_ids = _pad_ids(person_ids, num_bb)
        person_ids = torch.LongTensor(person_ids)
        boxes_mask = [1] * num_bb
        boxes_mask = torch.LongTensor(boxes_mask)

        return img_feat, img_pos_feat, origin_input_ids_q, corrupt_input_ids_q, origin_attn_masks, corrupt_attn_masks, origin_txt_type_ids, corrupt_txt_type_ids, swap_label, person_ids, boxes_mask  # , target, qid,


def vcr_itm_val_collate(inputs):
    (img_feats, img_pos_feats, origin_input_ids, corrupt_input_ids, origin_attn_masks, corrupt_attn_masks,
     origin_txt_type_ids, corrupt_txt_type_ids, swap_labels, person_ids, boxes_mask  # , target, qid,
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
    person_ids = pad_sequence(person_ids, batch_first=True, padding_value=0)
    boxes_mask = pad_sequence(boxes_mask, batch_first=True, padding_value=0)

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
             'person_ids': person_ids,
             'boxes_mask': boxes_mask
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
