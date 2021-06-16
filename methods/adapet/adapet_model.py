import numpy as np
import torch
from torch import nn

from methods.base_model import BaseModel

import log
logger = log.get_logger('root')

class AdaPETModel(BaseModel):
    def __init__(self, config, tokenizer, pvp):
        super(AdaPETModel, self).__init__(config, tokenizer, "mlm")
        self.config = config
        self.tokenizer = tokenizer
        self.pvp = pvp

        self.num_lbl = len(self.config.label_list)
        self.lbl_idx_lkup = nn.Embedding.from_pretrained(torch.eye(self.num_lbl)).cuda()


    def train_step(self, labeled_batch, extra_batch, alpha, **_):


        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        lbl_logits = self.get_single_logits(labeled_batch)

        reshape_lbl_logits = lbl_logits.reshape(-1)
        with torch.no_grad():
            lkup_lbl = self.lbl_idx_lkup(labels)
        reshape_lbl = lkup_lbl.reshape(-1)  # [bs*num_lbl]
        loss = torch.mean(nn.BCELoss()(reshape_lbl_logits, reshape_lbl))


        ### label_condition
        lm_inputs = self.generate_default_inputs(extra_batch)
        input_ids = extra_batch["original_input_ids"]
        mask_input_ids = extra_batch["input_ids"]
        tgt = extra_batch["tgt"]

        lm_logits = self.model(**lm_inputs)[0]
        lm_logits_vocab_prob = lm_logits.softmax(dim=-1)
        lm_logits_correct_vocab_prob = torch.gather(lm_logits_vocab_prob, 2, input_ids[:, :, None]).squeeze(2)
        max_seq_len = lm_logits_correct_vocab_prob.shape[1]

        mlm_loss = nn.BCELoss(reduce=False, reduction=None)
        full_loss = mlm_loss(lm_logits_correct_vocab_prob,
                             tgt[:, None].repeat(1, max_seq_len).float())

        mask_loss = input_ids != mask_input_ids

        pet_mlm_loss = torch.sum(full_loss * mask_loss.float()) / torch.max(torch.sum(mask_loss), torch.tensor(
            1).cuda())

        loss.backward()
        loss = alpha * loss.clone().detach() + (1 - alpha) * pet_mlm_loss

        return loss

    def get_single_logits(self, labeled_batch):

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        # TODO: only for single-token task
        logger.warning("only for single-token task")

        bs = labeled_batch["input_ids"].shape[0]
        max_num_lbl_tok = 1
        lbl_ids = np.ones((self.num_lbl, max_num_lbl_tok)) * self.tokenizer.pad_token_id
        list_lbl = [self.pvp.verbalize(item)[0] for item in self.config.label_list]
        for i, lbl in enumerate(list_lbl):
            i_lbl_ids = self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            assert len(i_lbl_ids) == 1
            lbl_ids[i, :len(i_lbl_ids)] = i_lbl_ids
        lbl_ids = torch.tensor(lbl_ids).cuda().long()

        list_mask_idx = np.ones((bs, 1)) * self.config.max_seq_length
        for bidx, idx in enumerate(mlm_labels):
            cur_mask_idx = mlm_labels[bidx].tolist().index(1)
            list_mask_idx[bidx, 0] = cur_mask_idx
        list_mask_idx = torch.tensor(list_mask_idx).cuda()

        init_mask_idx_lkup = torch.cat([torch.eye(self.config.max_seq_length), torch.zeros((1, self.config.max_seq_length))], dim=0)
        mask_idx_lkup = nn.Embedding.from_pretrained(init_mask_idx_lkup).cuda()
        with torch.no_grad():
            mask_idx_emb = mask_idx_lkup(list_mask_idx.long())

        pet_logits = self.model(**inputs)[0]
        pet_mask_logits = torch.matmul(mask_idx_emb[:, :, None, :], pet_logits[:, None, :, :]).squeeze(2)
        pet_mask_rep_vocab_prob = pet_mask_logits.softmax(dim=2)
        bs_by_max_num_lbl_tok = list(pet_mask_logits.shape[:2])

        mask_prob = torch.gather(pet_mask_rep_vocab_prob, 2, lbl_ids.view(1, -1).unsqueeze(1).repeat(bs_by_max_num_lbl_tok + [1]))
        mask_prob = mask_prob.transpose(1, 2)  # [bs,  max_num_lbl_tok*num_lbl, num_lbl_tok]
        mask_prob = mask_prob.reshape(bs, self.num_lbl, max_num_lbl_tok, max_num_lbl_tok)
        mask_diag_prob = torch.diagonal(mask_prob, dim1=2, dim2=3)  # [bs, num_lbl, num_lbl_tok]

        lbl_logits = torch.sum(mask_diag_prob, dim=2)
        return lbl_logits


    def eval_step(self, batch, **_):
        lbl_logits = self.get_single_logits(batch)
        pred_lbl, lbl_logits = torch.argmax(lbl_logits, dim=1), lbl_logits
        return lbl_logits


    def generate_default_inputs(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs