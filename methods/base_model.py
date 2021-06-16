from abc import abstractmethod

import torch
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, \
    AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    DebertaV2Tokenizer, DebertaV2Config

import log
from data_utils.preprocessor import SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER
from modified_hf_models.modeling_deberta_v2 import DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification

logger = log.get_logger('root')

MODEL_CLASSES = {
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
        MLM_WRAPPER: AlbertForMaskedLM,
    },
    'deberta': {
        'config': DebertaV2Config,
        'tokenizer': DebertaV2Tokenizer,
        MLM_WRAPPER: DebertaV2ForMaskedLM,
        SEQUENCE_CLASSIFIER_WRAPPER: DebertaV2ForSequenceClassification
    },
}



class BaseModel(torch.nn.Module):
    def __init__(self, config, tokenizer, wrapper_type):
        super(BaseModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list), finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)

        model_class = MODEL_CLASSES[self.config.model_type][wrapper_type]
        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)

        if "deberta" in self.config.model_name_or_path and self.config.fix_deberta:
            logger.info("fix_layers()")
            self.model.fix_layers()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                input_embeds=None, is_training=True, **kwargs):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          labels=labels,
                          input_embeds=input_embeds,
                          **kwargs)

    @abstractmethod
    def train_step(self, batch, extra_batch, alpha, **_):
        pass

    @abstractmethod
    def eval_step(self, batch, **_):
        pass

    @abstractmethod
    def generate_default_inputs(self, batch):
        pass
