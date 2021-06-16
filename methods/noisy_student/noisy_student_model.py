import copy

import torch

from methods.base_model import MODEL_CLASSES, BaseModel

import log
logger = log.get_logger('root')

class DropoutWords(torch.nn.Module):
    def __init__(self, drop_prob):
        super(DropoutWords, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs, is_training):
        assert len(inputs.shape) == 3
        # [batch_size, seq_len, embed_size]
        outputs = copy.deepcopy(inputs)
        if is_training and self.drop_prob > 0.0:
            dist = torch.distributions.Bernoulli(self.drop_prob)
            prob = dist.sample(sample_shape=(inputs.shape[0], inputs.shape[1]))
            outputs[prob == 1.0] = 0.0
            return outputs
        else:
            return outputs


class NoisyStudent(BaseModel):
    def __init__(self, config, tokenizer, pvp):
        super(NoisyStudent, self).__init__(config, tokenizer, "mlm")
        self.config = config
        self.tokenizer = tokenizer
        self.pvp = pvp

        self.dropout = DropoutWords(drop_prob=self.config.drop_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                input_embeds=None, is_training=True, **kwargs):

        raw_embeds = self.model.get_input_embeddings()(input_ids)  # [batch_size, seq_len, embed_size]
        if is_training:
            raw_embeds = self.dropout(raw_embeds, is_training=is_training)
        return self.model(inputs_embeds=raw_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          labels=labels, **kwargs)


    def train_step(self, batch, **_):
        inputs = self.generate_default_inputs(batch)
        mlm_labels, labels = batch['mlm_labels'], batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = torch.nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
        return loss

    def eval_step(self, batch, **_):
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def generate_default_inputs(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

