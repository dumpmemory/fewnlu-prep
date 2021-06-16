import torch

from methods.ptuning.prompt_encoder import PromptEncoder
from methods.base_model import BaseModel, MODEL_CLASSES

import log
logger = log.get_logger('root')

class ContinuousPromptModel(BaseModel):
    def __init__(self, config, tokenizer, pvp):
        super(ContinuousPromptModel, self).__init__(config, tokenizer, "mlm")
        self.config = config
        self.tokenizer = tokenizer
        self.prompt_encoder_head_type = config.prompt_encoder_head_type
        self.pattern_id = config.pattern_id
        self.prompt_length = self.pattern_id # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        self.pvp = pvp
        self.device = config.device

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False)

        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.embedding_size

        self.prompt_encoder = PromptEncoder(hidden_size=self.hidden_size,
                                            prompt_length=self.prompt_length,
                                            prompt_encoder_head_type=self.prompt_encoder_head_type,
                                            vocab_size=self.vocab_size,
                                            device=self.device,
                                            input_embeddings=self.model.get_input_embeddings())


    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):

        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          labels=labels,
                          **kwargs)


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

        input_ids = batch['input_ids']
        block_flags = batch["block_flags"]

        bz = batch['input_ids'].shape[0]
        model = self.model.module if hasattr(self.model, 'module') else self.model

        raw_embeds = model.model.get_input_embeddings()(input_ids)

        replace_embeds = model.prompt_encoder()
        replace_embeds = replace_embeds.unsqueeze(0) if len(replace_embeds.shape) == 1 else replace_embeds

        blocked_indices = (block_flags == -1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]
        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'deberta']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs