"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VCR
"""
import argparse
import json
import os
from os.path import abspath, dirname, exists, join
from time import time
from collections import defaultdict
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, Adamax
from model.adapter_layer import AdapterController
from apex import amp
from horovod import torch as hvd

from tqdm import tqdm
from data.itm_vcr_pg2 import * #(VcrItmDataset,VcrItmValDataset, vcr_itm_collate,vcr_itm_val_collate,VcrTxtTokLmdb)
from data import (TokenBucketSampler, PrefetchLoader, DetectFeatLmdb,
                  MetaLoader,
                  ImageLmdbGroup, ConcatDatasetWithLens,
                  VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate,)
from optim.adamw import RecAdam
from data.mlm_vcr_pg import (VcrMlmDataset, vcr_mlm_collate) #, VcrTxtTokLmdb)
from model.vcr_pretrain import UniterForVCRPretraining
from optim import AdamW, get_lr_sched

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

NUM_SPECIAL_TOKENS = 81

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def unfreeze_parameters(model, use_adapter):
    # targets = ['logit_fc'] ## ???
    # for n, p in model.named_parameters():
    #     if any(t in n for t in targets):
    #         p.required_grad = True
    #         print(f"{n} is trainable...")

    for name, sub_module in model.named_modules():
        if use_adapter:
            # if isinstance(sub_module, nn.LayerNorm):
            #     print(f"{name} is trainable...")
            #     # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
            #     for param_name, param in sub_module.named_parameters():
            #         param.requires_grad = True

            if isinstance(sub_module, AdapterController):
                print(f"{name} is trainable...")
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

            # if "vcr_itm_output" in name:
            #     print(f"{name} is trainable...")
            #     for param_name, param in sub_module.named_parameters():
            #         param.requires_grad = True

def print_trainable_params_percentage(model):
    orig_param_size = sum(p.numel() for p in model.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_size = count_parameters(model)

    percentage = trainable_size / orig_param_size * 100

    print(f"Trainable param percentage: {percentage:.2f}%")
    print(trainable_size)

    return percentage

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=batch_size, droplast=is_train)
        dataloader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=opts.n_workers, shuffle=False,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
        #dataloader = PrefetchLoader(dataloader)
    return dataloader

def build_vcr_dataset(is_train, opts, vcr_dbs, img_dbs):

    if is_train:
        LOGGER.info(f"build train vcr dataloaders")
        train_datasets = []
        train_datasets.append(
                VcrDataset(vcr_dbs[0], img_db_gt=img_dbs[0], img_db=img_dbs[1]))
        train_datasets.append(
                VcrDataset(vcr_dbs[1], img_db_gt=img_dbs[0], img_db=img_dbs[1]))
        dataset = ConcatDatasetWithLens(train_datasets)
        collate_fn = vcr_collate
    else:
        # val
        LOGGER.info(f"build val vcr dataloaders")
        LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
        val_txt_db = VcrTxtTokLmdb(opts.val_txt_db, -1)
        dataset = VcrEvalDataset("val", val_txt_db, img_db=img_dbs[1], img_db_gt=img_dbs[0])
        collate_fn = vcr_eval_collate
    return dataset, collate_fn

def build_vcr_mlm_dataset(is_train, opts, vcr_dbs, img_dbs):
    if is_train:
        LOGGER.info(f"build train mlm dataloaders")
        train_datasets = []
        train_datasets.append(
            VcrMlmDataset(vcr_dbs[0], img_db_gt=img_dbs[0], img_db=img_dbs[1], person_mask=opts.train_person_mask))
        train_datasets.append(
            VcrMlmDataset(vcr_dbs[1], img_db_gt=img_dbs[0], img_db=img_dbs[1], person_mask=opts.train_person_mask))

        dataset = ConcatDatasetWithLens(train_datasets)

    else:
        # val
        LOGGER.info(f"build val mlm dataloaders {opts.val_txt_db}, {opts.val_img_db}")
        ## ##
        val_dataset = []
        val_dataset.append(VcrMlmDataset(vcr_dbs[0], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        val_dataset.append(VcrMlmDataset(vcr_dbs[0], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        dataset = ConcatDatasetWithLens(val_dataset)
    collate_fn = vcr_mlm_collate
    return dataset, collate_fn


def build_vcr_itm_dataset(is_train, opts, vcr_dbs, img_dbs):
    person_corrupt = "person corrupt" if opts.train_person_corrupt else "random corrupt"

    if is_train:
        train_datasets = []
        LOGGER.info(f"build train itm dataloaders with {person_corrupt}")
        train_datasets.append(
            VcrItmDataset(vcr_dbs[0], img_db_gt=img_dbs[0], img_db=img_dbs[1], person_corrupt=opts.train_person_corrupt))
        train_datasets.append(
             VcrItmDataset(vcr_dbs[1], img_db_gt=img_dbs[0], img_db=img_dbs[1], person_corrupt=opts.train_person_corrupt))

        dataset = ConcatDatasetWithLens(train_datasets)
        collate_fn = vcr_itm_collate

    else:
        # val
        LOGGER.info(f"build val itm dataloaders {opts.val_txt_db}, {opts.val_img_db}")
        ## ##
        delta_dataset, acc_dataset = [], []

        delta_dataset.append(VcrItmValDataset(vcr_dbs[0], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        delta_dataset.append(VcrItmValDataset(vcr_dbs[1], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        delta_dataset = ConcatDatasetWithLens(delta_dataset)

        acc_dataset.append(VcrItmDataset(vcr_dbs[0], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        acc_dataset.append(VcrItmDataset(vcr_dbs[1], img_db=img_dbs[1], img_db_gt=img_dbs[0]))
        acc_dataset = ConcatDatasetWithLens(acc_dataset)
        dataset = (delta_dataset, acc_dataset)
        collate_fn = (vcr_itm_val_collate,vcr_itm_collate)
    return dataset, collate_fn

def create_dataloaders(multitask, is_train, opts):
    dataloaders = {}
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)

    #vcr_dbs = []
    #img_dbs = []

    if is_train:
        for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
            img_db, img_db_gt = load_img_feat(img_path, all_img_dbs, opts)
            qa_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qa")
            qar_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qar")
            vcr_dbs = [qa_txt_db, qar_txt_db]
            img_dbs = [img_db_gt, img_db]
    else:
        val_img_db, val_img_db_gt = load_img_feat(opts.val_img_db, all_img_dbs, opts)
        val_qa_txt_db = VcrTxtTokLmdb(opts.val_txt_db, opts.max_txt_len, task="qa")
        val_qar_txt_db = VcrTxtTokLmdb(opts.val_txt_db, opts.max_txt_len, task="qar")
        vcr_dbs = [val_qa_txt_db, val_qar_txt_db]
        img_dbs = [val_img_db_gt, val_img_db]

    for i, task in enumerate(multitask['tasks']):

        if task.startswith('vcr'):
            dataset, collate_fn = build_vcr_dataset(is_train, opts, vcr_dbs, img_dbs)

        elif task.startswith('itm'):
            if is_train:
                itm_qa_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qa")  # opts.train_itm_txt_dbs
                itm_qar_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qar")

                dataset, collate_fn = build_vcr_itm_dataset(is_train, opts, [itm_qa_txt_db, itm_qar_txt_db], img_dbs)

                #dataset, collate_fn = build_vcr_itm_dataset(is_train, opts, vcr_dbs, img_dbs)
            else:
                #val_itm_qar_txt_db = VcrTxtTokLmdb(opts.val_txt_db, opts.max_txt_len, task="qar")
                dataset, collate_fn = build_vcr_itm_dataset(is_train, opts, vcr_dbs, img_dbs)


        elif task.startswith("mlm"):
            dataset, collate_fn = build_vcr_mlm_dataset(is_train, opts, vcr_dbs, img_dbs)
        else:
            raise ValueError(f'Undefined task {task}')

        LOGGER.info(f"{len(dataset)*hvd.size()} samples loaded")

        if task.startswith('itm'):
            # itm handles distributed training in dset not sampler
            if is_train:
                loader = build_dataloader(dataset, collate_fn, is_train, opts)
            else:
                delta_dataset, acc_dataset = dataset
                vcr_itm_val_collate, vcr_itm_collate = collate_fn
                delta_loader = build_dataloader(delta_dataset, vcr_itm_val_collate, is_train, opts)
                acc_loader = build_dataloader(acc_dataset, vcr_itm_collate, is_train, opts)

                loader = (delta_loader, acc_loader)
        else:
            loader = build_dataloader(dataset,collate_fn, is_train, opts)

        if is_train:
            ratio = multitask['mix_ratio'][i]
            dataloaders[task] = (loader, ratio)
        else:
            if task.startswith("itm"):
                dataloaders[task] = [PrefetchLoader(loader[0]), PrefetchLoader(loader[1])]
            else:
                dataloaders[task] = PrefetchLoader(loader)

    return dataloaders




def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vcr_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vcr_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW

    elif opts.optim == "recadam":
        OptimCls = RecAdam
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_top.named_parameters() if
                           not any(nd in n for nd in no_decay) ],
                "weight_decay": opts.weight_decay,
                "anneal_w": opts.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in param_optimizer.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) ]
            },
            {
                "params": [p for n, p in param_top.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": opts.weight_decay,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in param_optimizer.named_parameters() if
                                    not any(nd in p_n for nd in no_decay)]
            },
            {
                "params": [p for n, p in param_top.named_parameters() if
                           any(nd in n for nd in no_decay) and opts.model_type in n],
                "weight_decay": 0.0,
                "anneal_w": opts.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in param_optimizer.named_parameters() if
                                    any(nd in p_n for nd in no_decay) ]
            },
            {
                "params": [p for n, p in param_top.named_parameters() if
                           any(nd in n for nd in no_decay) ],
                "weight_decay": 0.0,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in param_optimizer.named_parameters() if
                                    any(nd in p_n for nd in no_decay) ]
            }
        ]
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def load_img_feat(db_list, all_img_dbs, opts):
    db_ = db_list.split(";")
    assert len(db_) <= 2, "More than two img_dbs found"
    gt_db_path, db_path = "", ""
    for d in db_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
        all_img_dbs.path2imgdb[gt_db_path] = img_db_gt
    else:
        img_db_gt = None
    img_db = all_img_dbs[db_path] if db_path != "" else None
    all_img_dbs.path2imgdb[db_path] = img_db
    return img_db, img_db_gt


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # load DBs and image dirs
    #all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
    #                             opts.num_bb, opts.compressed_db)
    # build data loaders
    train_dataloaders = create_dataloaders(opts.multi_tasks, True, opts)
    print("after making train loader ")
    val_dataloader = create_dataloaders(opts.multi_tasks, False, opts)

    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)
    print("after making meta loader ")
    # Prepare model
    if opts.checkpoint and opts.checkpoint_from == "pretrain":
        ckpt = torch.load(opts.checkpoint)
        checkpoint = {k.replace('bert', 'uniter'): v for k, v in ckpt.items()}
    else:
        checkpoint = {}

    all_dbs = opts.train_txt_dbs + [opts.val_txt_db]
    #toker = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    #assert all(toker == json.load(open(f'{db}/meta.json'))['bert'] for db in all_dbs)

    model = UniterForVCRPretraining.from_pretrained(
        opts.model_config, checkpoint, img_dim=IMG_DIM, use_adapter=opts.use_adapter,num_special_tokens=NUM_SPECIAL_TOKENS)
    #if not opts.done_vcr_itm:
    #model.init_type_embedding()
    #

    if opts.checkpoint_from == "vcr_pretrain":
        ckpt = torch.load(opts.checkpoint)
        checkpoint = {k.replace('bert', 'uniter'): v for k, v in ckpt.items()}
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)

    model.init_word_embedding(1) # add itm-cls token
    model.cls.init_head(1)

    del checkpoint
    model.to(device)
    ### check # of parameter to train
    if opts.use_adapter:
        dfs_freeze(model)
        unfreeze_parameters(model, opts.use_adapter)
    percent_updated_parameters = print_trainable_params_percentage(model)

    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')
    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    #LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}') for task in train_dataloaders.keys()}
    # ITM w/ OT
    #if opts.itm_ot_lambda > 0:
    for task in train_dataloaders.keys():
        if task.startswith('itm'):
            task2loss[f'{task}_acc'] = RunningMeter(f'loss/{task}_acc')
            task2loss[f'{task}_delta'] = RunningMeter(f'loss/{task}_delta')

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    n_epoch = 0
    start = time()
    model.train()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    ##############################################################
    from transformers import BertTokenizer
    toker = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case='uncased')
    special_tokens_dict = {
        'additional_special_tokens': ["person_0", "person_1", "person_2", "person_3", "person_4",
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
                                      "person_75", "person_76", "person_77", "person_78", "person_79", "person_80"]}
    num_added_toks = toker.add_special_tokens(special_tokens_dict)
    ############################################
    while True:
        #validate(model, val_dataloader)
        for step, (task, batch) in enumerate(meta_loader):

            # if step < 10:
            #     input_ids = batch["input_ids"][0].cpu().tolist()
            #     #txt_labels = batch['txt_labels'][0]
            #     print(toker.convert_ids_to_tokens(input_ids))
            #     #print(txt_labels)
            #     print("--------------------------------------------------------------------")

            # forward pass
            n_examples[task] += batch['input_ids'].size(0)
            n_in_units[task] += (batch['attn_masks'] == 1).sum().item()
            loss = model(batch, task=task, compute_loss=True)
            n_loss_units[task] += torch.tensor(1) #loss.size(0)
            loss = loss.mean()  # loss is not normalized in model


            """
            label = batch["label"].long()
            label = torch.zeros(loader_params["batch_size"], 4).cuda().scatter_(1, label.unsqueeze(1), 1)# one hot , scatter(dim, index, src)

            visual_grad = torch.autograd.grad((output_dict["label_logits"] * (label.unsqueeze(1)>0).float()).sum(), batch["objects_feat"],create_graph=True)[0] # 정답 라벨의 확률에 영향을 준 probabilities
            # visual_grad shape : (bs, max_object_num, 2048)

            v_mask = torch.zeros(loader_params["batch_size"], batch["objects_feat"].shape[1]).cuda() # [bs max_object_num]
            visual_grad_cam = visual_grad.sum(2) # (bs, max_object_num)
            visual_mask = (batch["box_masks"]==0).bool() # [bs, max_object_num]
            visual_grad_cam = visual_grad_cam.masked_fill(visual_mask, -1e9)

             v_mask_pos, v_mask_neg = Select_obj_new_topn(visual_grad_cam, batch["box_masks"], top_num, loader_params["batch_size"], batch["objects_feat"].shape[1])

            
            # Grad Loss
            
            """


            # backward pass
            delay_unscale = ((step+1) % opts.gradient_accumulation_steps != 0)
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))
            task2loss[task](loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                print(step)
                global_step += 1
                lr_this_step = get_lr_sched(global_step, opts)
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.log_scaler_dict({l.name: l.val
                                           for l in task2loss.values()
                                           if l.val is not None})
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info(f'==============Step {global_step}===============')
                    for t in train_dataloaders.keys():
                        assert all(tt == t for tt in all_gather_list(t))
                        tot_ex = sum(all_gather_list(n_examples[t]))
                        ex_per_sec = int(tot_ex / (time() - start))
                        tot_in = sum(all_gather_list(n_in_units[t]))
                        in_per_sec = int(tot_in / (time() - start))
                        tot_l = sum(all_gather_list(n_loss_units[t]))
                        l_per_sec = int(tot_l / (time() - start))
                        LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                    f'{ex_per_sec} ex/s')
                        TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                             global_step)
                        TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                             global_step)
                        TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                             global_step)
                    LOGGER.info(f'===============================================')

                if global_step % opts.valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start validation')
                    validate(model, val_dataloader)
                    model_saver.save(model, global_step)

            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")
    LOGGER.info(f'**** Finished training. Step {global_step}: start validation ***')
    validate(model, val_dataloader)
    model_saver.save(model, global_step)

    with open(f'{opts.output_dir}/results/'
              f'results_{global_step}_final_qa_qar_'
              f'rank{rank}.json', 'w') as f:
        json.dump(results, f)


# def validate(model, val_dataloaders):
#     model.eval()
#     for task, loader in val_dataloaders.items():
#         LOGGER.info(f"validate on {task} task")
#         if task.startswith('mlm'):
#             val_log = validate_vcr_mlm(model, loader, "mlm")
#         elif task.startswith('vcr'):
#             val_log = validate_vcr(model, loader, "vcr")
#         elif task.startswith('itm'):
#             val_log = validate_vcr_itm(model, loader[0], loader[1], "itm")
#         else:
#             raise ValueError(f'Undefined task {task}')
#         val_log = {f'{task}_{k}': v for k, v in val_log.items()}
#         TB_LOGGER.log_scaler_dict(
#             {f'valid_{task}/{k}': v for k, v in val_log.items()})
#     model.train()


def validate(model, val_dataloaders):
    model.eval()
    # for task, loader in val_dataloaders.items():
    #     LOGGER.info(f"validate on {task} task")
    #     if task.startswith('mlm'):
    #         val_log = validate_vcr_mlm(model, loader, "mlm")
    #    elif task.startswith('vcr'):
    task = "vcr"
    val_log = validate_vcr(model, val_dataloaders[task], task)
        # elif task.startswith('itm'):
    #loader = val_dataloaders[task]
    #val_log = validate_vcr_itm(model, loader[0], loader[1], task)
        # else:
        #     raise ValueError(f'Undefined task {task}')
    val_log = {f'{task}_{k}': v for k, v in val_log.items()}
    TB_LOGGER.log_scaler_dict(
        {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_vcr_itm(model, delta_loader, acc_loader, task_adapter=None):
    LOGGER.info("start running VCR-ITM validation...")


    if hvd.rank() == 0:
        delta_pbar = tqdm(total=len(delta_loader))
        acc_pbar = tqdm(total=len(acc_loader))
    else:
        delta_pbar = NoOp()
        acc_pbar = NoOp()

    val_loss = 0
    delta_tot_score = 0
    acc_tot_score = 0

    delta_n_ex = 0
    acc_n_ex = 0
    results = {}
    st = time()
    for i, batch in enumerate(delta_loader):
        batch_origin = {'img_feat':batch['img_feat'],
                        'img_pos_feat':batch['img_pos_feat'],
                        'input_ids': batch['origin_input_ids'],
                        'txt_type_ids':batch['origin_txt_type_ids'],
                        'position_ids':batch['origin_position_ids'],
                        'attn_masks':batch['origin_attn_masks'],
                        'gather_index':batch['origin_gather_index'],
                        'person_ids': batch["person_ids"],
                        'boxes_mask': batch["boxes_mask"]
                        }

        batch_corrupt = {'img_feat': batch['img_feat'],
                        'img_pos_feat': batch['img_pos_feat'],
                        'input_ids': batch['corrupt_input_ids'],
                        'txt_type_ids': batch['corrupt_txt_type_ids'],
                        'position_ids': batch['corrupt_position_ids'],
                        'attn_masks': batch['corrupt_attn_masks'],
                        'gather_index': batch['corrupt_gather_index'],
                         'person_ids': batch["person_ids"],
                         'boxes_mask': batch["boxes_mask"]
                        }
        scores_origin = model(batch_origin, compute_loss=False, task=task_adapter)
        scores_corrupt = model(batch_corrupt, compute_loss=False, task=task_adapter)
        # print("!!!!!!!person corrupt!!!!!!!!!!!!!!!!!!!!!!!")
        # print(scores_corrupt)
        # print("!!!!!!!person original!!!!!!!!!!!!!!!!!!!!!!!")
        # print(scores_origin)
        targets = F.softmax(scores_origin, dim=1)[:, 1] - F.softmax(scores_corrupt, dim=1)[:,1]
        #targets = batch['targets']
        #qids = batch['qids']
        swap_targets = batch['swap_labels']
        swap_total = (swap_targets > 0).sum()
        # print("!!!!!!!!swap targets !!!!!!!!!!!!!!!!!!!!!!!")
        # print(swap_targets)
        # print("\n\n")
        if swap_total == 0:
            print("there is no person corrupted example")
            swap_score = torch.tensor(0)
        else:
            swap_score = torch.sum(targets[swap_targets > 0]) # /(swap_total + 1e-5)
        #print(swap_score)
        if torch.isnan(swap_score):
            swap_score = 0
            print("omg, nan is coming, change it to zero", swap_score)
            exit(0)
        delta_tot_score += swap_score
        delta_n_ex += swap_total
        delta_pbar.update(1)
    delta_tot_score = sum(all_gather_list(delta_tot_score))
    delta_n_ex = sum(all_gather_list(delta_n_ex))

    for i, batch in enumerate(acc_loader):
        scores = model(batch, compute_loss=False, task=task_adapter)
        targets = batch['targets']
        qids = batch['qids']

        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()
        acc_tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        acc_n_ex += len(targets)
        acc_pbar.update(1)

    val_loss = sum(all_gather_list(val_loss))
    acc_tot_score = sum(all_gather_list(acc_tot_score))
    acc_n_ex = sum(all_gather_list(acc_n_ex))

    tot_time = time()-st
    val_loss /= acc_n_ex
    val_acc = acc_tot_score / acc_n_ex
    val_delta = delta_tot_score / delta_n_ex

    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/delta': val_delta}

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"delta score: {val_delta:.2f}"
                f"acc score: {val_acc*100:.2f}"
                f"val loss : {val_loss :.2f}")
    return val_log #, results



@torch.no_grad()
def validate_vcr_mlm(model, val_loader, task_adapter=None):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()

    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, compute_loss=False, task=task_adapter)
        labels = batch['txt_labels']
        labels = labels[labels != -1]

        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
        val_pbar.update(1)

    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


@torch.no_grad()
def validate_vcr(model, val_loader, task_adapter=None):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()
    LOGGER.info("start running validation...")
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        scores = model(batch, compute_loss=False, task=task_adapter)
        qa_targets = batch['qa_targets']
        qar_targets = batch['qar_targets']
        qids = batch['qids']
        scores = scores[:, 1:]
        scores = scores.view(len(qids), -1)
        vcr_qa_loss = F.cross_entropy(
                scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
        if scores.shape[1] > 8:
            qar_index = [4+answer_ind.item()*4+i for answer_ind in qa_targets
                         for i in range(4)]
            qar_scores = scores[:, qar_index]
        else:
            qar_scores = scores[:, 4:]
        vcr_qar_loss = F.cross_entropy(
            qar_scores, qar_targets.squeeze(-1), reduction="sum")
        val_qa_loss += vcr_qa_loss.item()
        val_qar_loss += vcr_qar_loss.item()
        curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
            scores[:, :4], qa_targets, qar_scores, qar_targets)
        tot_qar_score += curr_qar_score
        tot_qa_score += curr_qa_score
        tot_score += curr_score
        for qid, score in zip(qids, scores):
            results[qid] = score.cpu().tolist()
        n_ex += len(qids)
        val_pbar.update(1)
    val_qa_loss = sum(all_gather_list(val_qa_loss))
    val_qar_loss = sum(all_gather_list(val_qar_loss))
    tot_qa_score = sum(all_gather_list(tot_qa_score))
    tot_qar_score = sum(all_gather_list(tot_qar_score))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/vcr_qa_loss': val_qa_loss,
               'valid/vcr_qar_loss': val_qar_loss,
               'valid/acc_qa': val_qa_acc,
               'valid/acc_qar': val_qar_acc,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score_qa: {val_qa_acc*100:.2f} "
                f"score_qar: {val_qar_acc*100:.2f} "
                f"score: {val_acc*100:.2f} ")
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument("--checkpoint_from",
                        default='pretrain', type=str,
                        choices=['pretrain', 'vcr_pretrain'],
                        help="which setting is checkpoint from")

    parser.add_argument("--use_adapter", action="store_true",
                        help="use adapter or no ")
    parser.add_argument("--train_person_mask", action="store_true",
                        help="use person corrupt or random corrupt")
    parser.add_argument("--train_person_corrupt", action="store_true",
                        help="use person corrupt or random corrupt")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='recadam',
                        choices=['adam', 'adamax', 'adamw', 'recadam'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=8,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    # adversarial training related
    parser.add_argument('--adv_training', action='store_true',
                        help="Whether to use adversarial training or not")
    parser.add_argument("--adv_modality", default=['text'],
                        help="add pertubation on text or image modality")
    parser.add_argument('--adv_lr_txt', type=float, default=0)
    parser.add_argument('--adv_lr_img', type=float, default=0)
    parser.add_argument('--adv_steps', type=int, default=1,
                        help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2",
                        choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0,
                        help="set to 0 to be unlimited")
    parser.add_argument('--adv_kl_weight', type=float, default=0,
                        help="set to 0 to be unlimited")


    # RecAdam parameters
    parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'],
                        help="the type of annealing function in RecAdam. Default sigmoid")
    parser.add_argument("--recadam_anneal_k", type=float, default=0.5, help="k for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_t0", type=int, default=250, help="t0 for the annealing function in RecAdam.")
    parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                        help="Weight for the annealing function in RecAdam. Default 1.0.")
    parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                        help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")

    parser.add_argument("--logging_Euclid_dist", action="store_true",
                        help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
    parser.add_argument("--start_from_pretrain", action="store_true",
                        help="Whether to initialize the model with pretrained parameters")

    parser.add_argument("--albert_dropout", default=0.0, type=float,
                        help="The dropout rate for the ALBERT model")

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
