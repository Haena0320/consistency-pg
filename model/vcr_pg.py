"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
from collections import defaultdict

from torch import nn
import torch
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from copy import deepcopy
# from .layer import GELU
from .model import (UniterPreTrainedModel, UniterModel)

class UniterForVCRImageTextMatching(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim, use_adapter):
        super().__init__(config, img_dim, use_adapter)
        self.uniter = UniterModel(config, img_dim, use_adapter)
        self.vcr_itm_output = nn.Linear(config.hidden_size, 3) # for AUM 
        self.contrastive_person_swap = False #config.contrastive_person_swap
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, adv_training=False, adv_modality=None,
                adv_delta_txt=None, adv_delta_img=None, compute_loss=True, task=None):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        person_ids = batch['person_ids']
        boxes_mask = batch['boxes_mask']
        targets = batch["targets"]

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids, person_ids=person_ids, boxes_mask=boxes_mask,
                                      task=[task])

        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.vcr_itm_output(pooled_output)

        sp_loss = 0
        if self.contrastive_person_swap:
            cams = [self.uniter.encoder.layer[i].attention.self.get_attn() for i in range(12)]
            cams = torch.stack(cams, dim=1).max(1)[0].max(1)[0]

            gather_person_index = deepcopy(batch['person_masks'])
            gather_person_index[gather_person_index < 0] = 0

            gather_object_index_neg = deepcopy(batch['object_masks'])
            gather_object_index_neg[gather_object_index_neg < 4] = 0

            person_swap_bool = torch.sum(gather_object_index_neg > 0, dim=-1)
            gather_object_index_neg = gather_object_index_neg[person_swap_bool > 0]
            neg = torch.sum(cams[gather_person_index > 0] * gather_object_index_neg) / torch.sum(person_swap_bool)

            gather_object_index_pos = deepcopy(batch['object_masks'])
            gather_object_index_pos[gather_object_index_pos != 3] = 0
            gather_object_index_pos[gather_object_index_pos == 3] = 1

            batch_object_num = gather_object_index_pos.sum(1)
            pos = torch.mean(cams[gather_person_index > 0]*gather_object_index_pos[batch_object_num > 0], dim=-1)[0].sum() / torch.sum(person_swap_bool)
            sp_loss = max(torch.tensor(0).half().cuda(), -neg + pos + 0.1)

        if compute_loss:
            vcr_itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return vcr_itm_loss + 0.2 * sp_loss
            #return sp_loss
        else:
            return itm_scores

