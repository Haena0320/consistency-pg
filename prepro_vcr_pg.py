"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from transformers import BertTokenizer
from data.data import open_lmdb
import numpy as np
from copy import deepcopy
import random

#txt_dump = {'input_ids': [1184, 1110, 28996, 2296, 1268, 1208, 136], 'input_ids_masks': [-1, -1, 0, -1, -1, -1, -1], 'input_ids_as': ([1119, 1110, 2296, 15546, 119], [1119, 1110, 4853, 1105, 1169, 1136, 2059, 1184, 1119, 1110, 4510, 28996, 1587, 1140, 119], [1119, 1110, 2296, 2785, 5528, 119], [1119, 1110, 2296, 12003, 119]), 'input_ids_as_masks': ([-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]), 'qa_target': 3, 'input_ids_rs': ([1119, 1144, 170, 2003, 1113, 1117, 1339, 119], [28996, 1110, 1702, 2626, 1120, 28996, 10573, 1158, 1106, 170, 13952, 12545, 1105, 15861, 1158, 1272, 1119, 6191, 1103, 3943, 1110, 6276, 119], [1128, 1169, 1587, 1118, 1117, 2838, 1105, 1404, 1846, 119], [1119, 1144, 23106, 1440, 1113, 1117, 1339, 119]), 'input_ids_rs_masks': ([-1, -1, -1, -1, -1, -1, -1, -1], [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]), 'qar_target': 0, 'img_fname': ('vcr_gt_val_3038_ITS_COMPLICATED_00.08.14.616-00.08.17.313@0.npz', 'vcr_val_3038_ITS_COMPLICATED_00.08.14.616-00.08.17.313@0.npz')}


# model.resize_token_embeddings(len(toker))
#toker = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
# text_list = [[1], "is", "apprehensive", "about", "her", "coming", "."]
# name = ['person', "person", "person"]
# toker.tokenize("apprehensive")
# toker.convert_tokens_to_ids(["he", "is", 'app', '##re', '##hen', '##sive'])
# ids, _ = bert_tokenize(toker, text_list, name)
# toker.convert_ids_to_tokens(ids)

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

person_candidates = list(range(len(additional_special_tokens)))
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
                    try:
                        if name[int(e)] == "person":
                            tmp = name[int(e)] + f'_{e}'
                            person_masks.append(int(e))
                        else:
                            tmp = name[int(e)]
                            person_masks.append(-1)
                    except:
                        tmp = f"person_{e}"
                        person_masks.append(int(e))

                new_text.append(tmp)

        else:
            tokens = tokenizer.tokenize(word)
            new_text.extend(tokens)
            person_masks.extend([-1]*len(tokens))
    ids = tokenizer.convert_tokens_to_ids(new_text)
    assert len(ids) == len(person_masks)
    return ids, person_masks

names = ["person", "person", "bicycle", "tie", "clock"]
question = ["Where", "did", [0], "come", "from", "before", "sitting", "on", "the", "bed", "with", [1], "?"]


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


# import random
# random.choice(list[1,2,3,4])

def change_person_prior(data_one, person_candidates):
    input_all = data_one["objects"]
    rest_list = sorted(list(filter(lambda x: input_all[x] == "person", range(len(input_all)))))  # person index 만
    answer_label = data_one["answer_label"]
    rationale_label = data_one["rationale_label"]
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
                    new_answer = change_person_priors_in_choice(data_one["answer_choices"][answer_label], tok, tok_orig)
                    new_rationale = change_person_priors_in_choice(data_one["rationale_choices"][rationale_label], tok,
                                                                   tok_orig)
                elif len(candidates) == 0:
                    print("no person last in image, let's choose one not in image ")
                    not_in_target_image = True
                    person_candidates = list(set(person_candidates) - set(tok))
                    tok = random.sample(person_candidates, len(tok))
                    assert tok != tok_orig
                    new_answer = change_person_priors_in_choice(data_one['answer_choices'][answer_label], tok, tok_orig)
                    new_rationale = change_person_priors_in_choice(data_one['rationale_choices'][rationale_label], tok,
                                                                   tok_orig)
            else:
                if cng_tok == tok:
                    tok = deepcopy(tok_orig)

        new_question.append(tok)

    assert len(new_answer) != 0
    assert len(new_question) != 0
    return new_question, new_answer, new_rationale, not_in_target_image

def swap_person_input_ids(input_ids, input_ids_masks, names, toker):
    person_list = list(filter(lambda x: x != -1, input_ids_masks))
    swap_person_idx = input_ids_masks.index(person_list[0])

    person_in_images = list(filter(lambda x : names[x] == "person", range(len(names))))
    swap_candidates = list(set(person_in_images)-set(person_list))
    swap_candidate = list(np.random.choice(swap_candidates, 1))
    swapped_input_ids = deepcopy(input_ids)
    swapped_input_ids[swap_person_idx] = toker.convert_tokens_to_ids(f"person_{swap_candidate[0]}")
    return swapped_input_ids

def process_vcr(jsonl, db, tokenizer, toker, person_candidates, missing=None, split="train"):
    id2len_qa = {}
    id2len_qar = {}
    txt2img = {}  # not sure if useful
    img2txts = {} # not sure if useful
    debug_ = 0
    for line in tqdm(jsonl, desc='processing VCR with person'):
        example_ = {}
        example = json.loads(line)
        id_ = example["annot_id"]
        img_id = example["metadata_fn"].split('/')[-1][:-5]
        img_fname = (f'vcr_gt_{split}_{img_id}.npz', f'vcr_{split}_{img_id}.npz')
        img2txt_key = f'vcr_gt_{split}_{img_id}.npz'
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        txt2img[id_] = img_fname
        if img2txt_key in img2txts:
            img2txts[img2txt_key].append(id_)
        else:
            img2txts[img2txt_key] = [id_]

        names = example["objects"]
        input_ids, input_ids_masks = tokenizer(example['question'], names)
        input_ids_as, input_ids_as_masks = zip(*[tokenizer(answer, names) for answer in example["answer_choices"]])


        person_list = list(filter(lambda x : x != -1, input_ids_masks))

        example_["person_negative"] = len(person_list)
        example_['objects'] = names
        example_['question'] = example['question']
        example_['answer_gt'] = example['answer_choices'][example['answer_label']]
        example_['rationale_gt'] = example['rationale_choices'][example['rationale_label']]

        input_ids_rs, input_ids_rs_masks = zip(
            *[tokenizer(rationale, names) for rationale in example["rationale_choices"]])
        if len(person_list) > 0:
            swapped_question, swapped_answer, swapped_rationale, not_in_target_image = change_person_prior(example, person_candidates)
            assert len(swapped_answer) != 0
            person_negative_input_ids, person_negative_input_ids_masks = tokenizer(swapped_question, names, not_in_target_image)
            person_negative_input_ids_gt, person_negative_input_ids_gt_masks = tokenizer(swapped_answer, names, not_in_target_image)
            person_negative_input_ids_rs_gt, person_negative_input_ids_rs_gt_masks = tokenizer(swapped_rationale, names, not_in_target_image)
            example_['person_negative_input_ids'] = person_negative_input_ids
            example_['person_negative_input_ids_masks'] = person_negative_input_ids_masks

            example_['person_negative_input_ids_gt'] = person_negative_input_ids_gt
            example_['person_negative_input_ids_gt_masks'] = person_negative_input_ids_gt_masks

            example_['person_negative_input_ids_rs_gt'] = person_negative_input_ids_rs_gt
            example_['person_negative_input_ids_rs_gt_masks'] = person_negative_input_ids_rs_gt_masks


                # print("!!!!!!!!!!!!!!!swap person question !!!!!!!!!!!!!!")
                # print(toker.convert_ids_to_tokens(person_negative_input_ids))
                # print("!!!!!!!!!!!!!!!swap gt answer !!!!!!!!!!!!!!")
                # print(toker.convert_ids_to_tokens(person_negative_input_ids_gt))
                # print("!!!!!!!!!!!!!!!swap gt rationale !!!!!!!!!!!!!!")
                # print(toker.convert_ids_to_tokens(person_negative_input_ids_rs_gt))
                # print("\n\n")
                # print("\n\n")
            assert len(person_negative_input_ids_gt) != 0
            assert len(person_negative_input_ids_rs_gt) != 0


        if debug_ < 10:
            print("!!!!!!!!!!!!!!!original person question !!!!!!!!!!!!!!")
            print(toker.convert_ids_to_tokens(input_ids))
            print("!!!!!!!!!!!!!!!original gt answer !!!!!!!!!!!!!!")
            print(toker.convert_ids_to_tokens(input_ids_as[0]))
            print(toker.convert_ids_to_tokens(input_ids_as[1]))
            print(toker.convert_ids_to_tokens(input_ids_as[2]))
            print(toker.convert_ids_to_tokens(input_ids_as[3]))
            print("!!!!!!!!!!!!!!!original  gt rationale !!!!!!!!!!!!!!")
            print(toker.convert_ids_to_tokens(input_ids_rs[0]))
            print(toker.convert_ids_to_tokens(input_ids_rs[1]))
            print(toker.convert_ids_to_tokens(input_ids_rs[2]))
            print(toker.convert_ids_to_tokens(input_ids_rs[3]))
            print("\n\n")

        id2len_qa[id_] = len(input_ids)
        example_['input_ids'] = input_ids
        example_['input_ids_masks'] = input_ids_masks

        example_["input_ids_as"] = input_ids_as
        example_["input_ids_as_masks"] = input_ids_as_masks
        example_['qa_target'] = example["answer_label"]


        example_["input_ids_rs"] = input_ids_rs
        example_["input_ids_rs_masks"] = input_ids_rs_masks
        example_['qar_target'] = example["rationale_label"]

        id2len_qar[id_] = len(input_ids) + len(input_ids_as[example["answer_label"]]) + 1
        example_['img_fname'] = img_fname
        db[id_] = example_
        debug_ +=1
        if debug_ < 20:
            print(example_)
            print("-------------------------------")

    return id2len_qa, id2len_qar, txt2img, img2txts





def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    # toker = BertTokenizer.from_pretrained(
    #    opts.toker, do_lower_case='uncased' in opts.toker)
    toker = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
    special_tokens_dict = {
        'additional_special_tokens': additional_special_tokens}
    num_added_toks = toker.add_special_tokens(special_tokens_dict)

    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids(['!'])[0],
                       len(toker))
    meta["num_added_toks"] = num_added_toks
    meta["person0"] = toker.convert_tokens_to_ids(['person_0'])[0]
    meta["person19"] = toker.convert_tokens_to_ids(['person_19'])[0]
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)


    with open(f'{opts.output}/special_token.json', 'w') as f:
        json.dump(special_tokens_dict["additional_special_tokens"], f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation) as ann:
            if opts.missing_imgs is not None:
                missing_imgs = set(json.load(open(opts.missing_imgs)))
            else:
                missing_imgs = None
            id2len_qa,id2len_qar, txt2img, img2txts = process_vcr(ann, db, tokenizer, toker, person_candidates, missing_imgs, opts.split)


    with open(f'{opts.output}/id2len_qa.json', 'w') as f:
        json.dump(id2len_qa, f)
    with open(f'{opts.output}/id2len_qar.json', 'w') as f:
        json.dump(id2len_qar, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/img2txts.json', 'w') as f:
        json.dump(img2txts, f)


if __name__ == '__main__':
    # train, val, test,  * qa, qar -> 6번
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--split', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB, [vcr_test.db, vcr_train.db, vcr_test.db]')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)






