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
from .layer import GELU, BertOnlyMLMHead
from .model import (UniterPreTrainedModel, UniterModel)

class UniterForVCRPretraining(UniterPreTrainedModel):
    """  Person-focused Pretrain UNITER for VCR
    """
    def __init__(self, config, img_dim, use_adapter, num_special_tokens):
        super().__init__(config, img_dim, use_adapter)
        self.uniter = UniterModel(config, img_dim, use_adapter)
        self.init_type_embedding()
        self.init_word_embedding(num_special_tokens)
        self.cls = BertOnlyMLMHead(config, self.uniter.embeddings.word_embeddings.weight)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(config.hidden_size * 2, eps=1e-12),
            nn.Linear(config.hidden_size * 2, 2)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.m_rate = config.m_rate # [0.5, 2, 4]
        self.kl_rate = config.kl_rate # [0.5, 1]
        self.maxg = config.maxg #[0,2]

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
        new_emb = nn.Embedding(orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb
        print("emb", new_emb.weight.size())

    def forward(self, batch, adv_training=False, adv_modality=None,
                adv_delta_txt=None, adv_delta_img=None, compute_loss=True, task=None):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        txt_type_ids = batch['txt_type_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        person_ids = batch['person_ids']
        boxes_mask = batch['boxes_mask']

        if task=="mlm":
            targets = batch["txt_labels"]
            return self.forward_mlm(input_ids, position_ids, img_feat, img_pos_feat,
                        attn_masks, adv_training, adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index, txt_type_ids,targets, person_ids, boxes_mask, task, compute_loss)

        elif task =="itm":
            targets = batch['targets']
            return self.forward_itm(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,txt_type_ids, targets,person_ids, boxes_mask, task, compute_loss)

        elif task == "vcr":
            targets = batch['targets']
            return self.forward_vcr(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,txt_type_ids,targets,person_ids, boxes_mask, task, compute_loss)


        elif task =="vcr_tdt":
            targets = batch['targets']
            sequence_output, positive_sequence_output, negative_sequence_output, knowledge_mask = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, adv_training,
                                          adv_modality, adv_delta_txt,
                                          adv_delta_img, gather_index,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=txt_type_ids, person_ids=person_ids, boxes_mask=boxes_mask, task=[task])

            pooled_output = self.uniter.pooler(sequence_output)
            positive_pooled_output = self.uniter.pooler(positive_sequence_output)
            negative_pooled_output = self.uniter.pooler(negative_sequence_output)

            pooled_output = self.dropout(pooled_output)
            positive_pooled_output = self.dropout(positive_pooled_output)
            negative_pooled_output = self.dropout(negative_pooled_output)

            logits = self.vcr_output(pooled_output)
            mask_logits = self.vcr_output(positive_pooled_output)
            negative_logits = self.vcr_output(negative_pooled_output)

            if compute_loss:
                vcr_loss = F.cross_entropy(
                    logits, targets.squeeze(-1),
                    reduction='mean')

                mask_loss = F.cross_entropy(
                    mask_logits, targets.squeeze(-1),
                    reduction='mean')

                loss_fct = nn.KLDivLoss()
                kl_loss = max(0.,
                              loss_fct(torch.log_softmax(mask_logits, dim=-1), torch.softmax(logits, dim=-1)) - \
                              0.5 * loss_fct(torch.log_softmax(negative_logits, dim=-1),
                                             torch.softmax(mask_logits, dim=-1)) - \
                              0.5 * loss_fct(torch.log_softmax(negative_logits, dim=-1), torch.softmax(logits, dim=-1)) \
                              + self.maxg)

                tot_loss = vcr_loss + self.kl_rate * kl_loss + self.m_rate * mask_loss
                return tot_loss

            return logits


        elif task =="itm_tdt":
            targets = batch['targets']
            sequence_output, positive_sequence_output, negative_sequence_output, knowledge_mask = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, adv_training,
                                          adv_modality, adv_delta_txt,
                                          adv_delta_img, gather_index,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=txt_type_ids,person_ids=person_ids, boxes_mask=boxes_mask, task=[task])

            pooled_output = self.uniter.pooler(sequence_output)
            positive_pooled_output = self.uniter.pooler(positive_sequence_output)
            negative_pooled_output = self.uniter.pooler(negative_sequence_output)

            pooled_output = self.dropout(pooled_output)
            positive_pooled_output = self.dropout(positive_pooled_output)
            negative_pooled_output = self.dropout(negative_pooled_output)

            logits = self.itm_output(pooled_output)
            mask_logits = self.itm_output(positive_pooled_output)
            negative_logits = self.itm_output(negative_pooled_output)

            if compute_loss:
                itm_loss = F.cross_entropy(logits, targets, reduction='none')

                mask_loss = F.cross_entropy(
                    mask_logits, targets.squeeze(-1),
                    reduction='none')

                loss_fct = nn.KLDivLoss()
                kl_loss = max(0.,
                              loss_fct(torch.log_softmax(mask_logits, dim=-1), torch.softmax(logits, dim=-1)) - \
                              0.5 * loss_fct(torch.log_softmax(negative_logits, dim=-1),
                                             torch.softmax(mask_logits, dim=-1)) - \
                              0.5 * loss_fct(torch.log_softmax(negative_logits, dim=-1), torch.softmax(logits, dim=-1)) \
                              + self.maxg)

                tot_loss = itm_loss + self.kl_rate * kl_loss + self.m_rate * mask_loss
                return tot_loss

            return logits

    def forward_itm(self,input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,txt_type_ids, targets,
                                      person_ids, boxes_mask, task, compute_loss=True):


        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,person_ids=person_ids,
                                      boxes_mask=boxes_mask, task=[task])

        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        if compute_loss:
            vcr_itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return vcr_itm_loss
        else:
            return itm_scores

    def forward_vcr(self, input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,txt_type_ids,targets,
                                      person_ids,boxes_mask, task, compute_loss=True):

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids, person_ids=person_ids,boxes_mask=boxes_mask, task=[task])

        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)

        if compute_loss:
            vcr_loss = F.cross_entropy(
                rank_scores, targets.squeeze(-1),
                reduction='mean')
            return vcr_loss
        else:
            return rank_scores

    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                        attn_masks, adv_training, adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,txt_type_ids,txt_labels, person_ids,boxes_mask,task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, adv_training,
                                      adv_modality, adv_delta_txt,
                                      adv_delta_img, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,person_ids=person_ids, boxes_mask=boxes_mask, task=[task])

        # get only the txt part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            vcr_masked_lm_loss = F.cross_entropy(prediction_scores, txt_labels[txt_labels != -1], reduction='none')
            return vcr_masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

