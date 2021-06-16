# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import copy
import random

import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import Sampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, \
    AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    DebertaV2Tokenizer, DebertaV2Config

#, DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification, DebertaV2Config

# TODO: for deberta-v2
from configs import WrapperConfig
from methods.adapet.adapet_model import AdaPETModel
from methods.lm_training.lm_training_model import LMTrainingModel
from methods.sequence_classifier.cls_model import SequenceClassifierModel
from modified_hf_models.modeling_deberta_v2 import DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification

from transformers import __version__ as transformers_version
from transformers.data.metrics import simple_accuracy

import log
from data_utils import preprocessor
from methods.task_helpers import TASK_HELPERS
from data_utils.preprocessor import MLM_WRAPPER, SEQUENCE_CLASSIFIER_WRAPPER, PLM_WRAPPER, PREPROCESSORS
from methods.base_model import MODEL_CLASSES
from methods.pet.pet_model import PetModel
from methods.ptuning.ptuning_model import ContinuousPromptModel
from methods.utils import InputFeatures, DictDataset, distillation_loss, exact_match, get_verbalization_ids


logger = log.get_logger('root')
CONFIG_NAME = 'wrapper_config.json'


METHOD_CLASSES = {
    'sequence_classifier': SequenceClassifierModel,
    'pet': PetModel,
    'ptuning': ContinuousPromptModel,
    'adapet': AdaPETModel,
    'lm_training': LMTrainingModel,
    'ipet': PetModel
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
    # MARGIN_MLM_WRAPPER: lambda wrapper: wrapper.margin_mlm_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
    # MARGIN_MLM_WRAPPER: lambda wrapper: wrapper.margin_mlm_train_step,
}






# todo
# Re-implement RandomSampler to decouple independent random seed.
class RandomSampler(Sampler):
    def __init__(self, data_source, seed) -> None:
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.seed = seed

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)



class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""
    def __init__(self, config: WrapperConfig, pattern_id):
        self.config = config
        self.config.pattern_id = pattern_id

        tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        # if self.config.model_type == 'gpt2':
        #     self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token

        self.wrapper_type = "cls" if config.method == "sequence_classifier" else "mlm"
        self.preprocessor = PREPROCESSORS[self.wrapper_type](self, task_name=config.task_name,
                                                                    pattern_id=pattern_id,
                                                                    verbalizer_file=config.verbalizer_file,
                                                                    use_continuous_prompt=config.use_continuous_prompt)

        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None



        if config.method == "sequence_classifier":
            assert config.use_cloze == False
            self.model = SequenceClassifierModel(config, self.tokenizer)

        elif config.method == "pet":
            assert config.use_cloze == True and config.use_continuous_prompt == False
            self.model = PetModel(config, self.tokenizer, self.preprocessor.pvp)

        elif config.method == "ptuning":
            assert config.use_cloze == True and config.use_continuous_prompt == True
            self.model = ContinuousPromptModel(config, self.tokenizer, self.preprocessor.pvp)

        elif config.method == "noisy_student":
            pass
        elif config.method == "adapet":
            assert config.use_cloze == True and config.use_continuous_prompt == False
            self.model = AdaPETModel(config, self.tokenizer, self.preprocessor.pvp)
        elif config.method == "lm_training":
            assert config.use_cloze == True and config.use_continuous_prompt == False
            self.model = LMTrainingModel(config, self.tokenizer, self.preprocessor.pvp)
        elif config.method == "ipet":
            assert config.use_cloze == True and config.use_continuous_prompt == False
            self.model = PetModel(config, self.tokenizer, self.preprocessor.pvp)
        else:
            raise NotImplementedError(f"Training method '{config.method}' not implemented.")

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()


    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)

        wrapper_type = "cls" if wrapper.config.method == "sequence_classifier" else "mlm"
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper_type]

        wrapper.preprocessor = PREPROCESSORS[wrapper_type](wrapper,
                                                          task_name=wrapper.config.task_name,
                                                          pattern_id=wrapper.config.pattern_id,
                                                          use_continuous_prompt=wrapper.config.use_continuous_prompt)


        if wrapper.config.method == "pet":
            wrapper.model = PetModel(wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
            wrapper.model.model = model_class.from_pretrained(path)
        elif wrapper.config.method == "ptuning":
            wrapper.model = ContinuousPromptModel(wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
            wrapper.model.model = model_class.from_pretrained(path)
            save_path_file = os.path.join(path, "embeddings.pth")
            data = torch.load(save_path_file)
            wrapper.model.prompt_encoder.load_state_dict(data["prompt_encoder"])
        elif wrapper.config.method == "sequence_classifier":
            wrapper.model = SequenceClassifierModel(wrapper.config, wrapper.tokenizer)
            wrapper.model.model = model_class.from_pretrained(path)
        elif wrapper.config.method == 'adapet':
            wrapper.model = AdaPETModel(wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
            wrapper.model.model = model_class.from_pretrained(path)
        elif wrapper.config.method == "lm_training":
            wrapper.model = LMTrainingModel(wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
            wrapper.model.model = model_class.from_pretrained(path)
        elif wrapper.config.method == "ipet":
            wrapper.model = PetModel(wrapper.config, wrapper.tokenizer, wrapper.preprocessor.pvp)
            wrapper.model.model = model_class.from_pretrained(path)
        else:
            raise NotImplementedError()


        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None

        if torch.cuda.device_count() > 1:
            wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()
        return wrapper


    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        if not self.config.use_continuous_prompt:
            return

        state = {
            "prompt_encoder": model_to_save.prompt_encoder.state_dict()
        }
        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)
        return

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, train_data, dev32_data, pattern_iter_output_dir, train_eval_config, unlabeled_data, ipet_train_data):

        results_dict = {}
        all_train_data = []
        if ipet_train_data is not None:
            train_data = train_data + ipet_train_data

        if train_eval_config.train_priming:
            pass
        else:
            all_train_data = train_data

        """
        if ipet_train_data is not None:
            train_data = train_data + ipet_train_data
        
        # TODO: train_priming
        all_train_data = []
        all_unlabeled_data = []

        priming_num = config.priming_num
        """
        """
        if config.priming:
            for i, example in enumerate(train_data):
                all_priming_data = train_data.copy()
                all_priming_data.remove(example)

                priming_example = random.sample(all_priming_data, k=priming_num)
                example.meta['priming_data'] = priming_example
                all_train_data.append(example)

            if unlabeled_data is not None:
                for i, example in enumerate(unlabeled_data):
                    all_priming_unlabel = unlabeled_data.copy()
                    all_priming_unlabel.remove(example)
                    priming_unlabeled_example = random.sample(all_priming_unlabel, k=priming_num)
                    example.meta["priming_data"] = priming_unlabeled_example
                    all_unlabeled_data.append(example)

        else:
            logger.info("not priming during training.")
            all_train_data = train_data
            all_unlabeled_data = unlabeled_data
        """

        global_step, tr_loss = self._train(train_data=all_train_data,
                                           dev32_data=dev32_data,
                                           unlabeled_data=unlabeled_data,
                                           per_gpu_train_batch_size=train_eval_config.per_gpu_train_batch_size,
                                           per_gpu_eval_batch_size=train_eval_config.per_gpu_eval_batch_size,
                                           per_gpu_unlabeled_batch_size=train_eval_config.per_gpu_unlabeled_batch_size,
                                           learning_rate=train_eval_config.learning_rate,
                                           embedding_learning_rate=train_eval_config.embedding_learning_rate,
                                           weight_decay=train_eval_config.weight_decay,
                                           adam_epsilon=train_eval_config.adam_epsilon,
                                           warmup_step_ratio=train_eval_config.warmup_step_ratio,
                                           gradient_accumulation_steps=train_eval_config.gradient_accumulation_steps,
                                           max_grad_norm=train_eval_config.max_grad_norm,
                                           train_priming=train_eval_config.train_priming,
                                           sampler_seed=train_eval_config.sampler_seed,
                                           eval_priming=train_eval_config.eval_priming,
                                           priming_num=train_eval_config.priming_num,
                                           max_steps=train_eval_config.max_steps,
                                           num_train_epochs=train_eval_config.num_train_epochs,
                                           metrics=train_eval_config.metrics,
                                           decoding_strategy=train_eval_config.decoding_strategy,
                                           alpha=train_eval_config.alpha,
                                           temperature=train_eval_config.temperature,
                                           early_stop_epoch=train_eval_config.early_stop_epoch,
                                           every_eval_step=train_eval_config.every_eval_step,
                                           pattern_iter_output_dir=pattern_iter_output_dir,
                                           n_gpu=train_eval_config.n_gpu, device=train_eval_config.device)
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss
        return results_dict


    def evaluate(self, eval_data, per_gpu_eval_batch_size, n_gpu, device, metrics, decoding_strategy, eval_priming,
                 priming_num, priming_data=None):
        all_eval_data = []
        if eval_priming and priming_data:
            for example in eval_data:
                priming_example = random.sample(priming_data, k=priming_num)
                example.meta['priming_data'] = priming_example
                all_eval_data.append(example)
        else:
            all_eval_data = eval_data

        self.model.eval()
        results = self._eval(all_eval_data,
                             device,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             n_gpu=n_gpu,
                             decoding_strategy=decoding_strategy,
                             eval_priming=eval_priming,
                             priming_num=priming_num)

        predictions = np.argmax(results['logits'], axis=1)
        scores = {}

        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise ValueError(f"Metric '{metric}' not implemented")

        results['scores'] = scores
        results['predictions'] = predictions
        return results


    def _prepare_dataloader(self, type, data, per_gpu_batch_size, use_priming, sampler_seed, n_gpu, labelled):
        batch_size = per_gpu_batch_size * max(1, n_gpu)
        dataset = self._generate_dataset(data, priming=use_priming, labelled=labelled)

        if type == "train" or type == "extra":
            sampler = RandomSampler(dataset, sampler_seed)
        elif type == 'dev32' or type == "eval":
            sampler = SequentialSampler(dataset)
        else:
            raise NotImplementedError("Type {} not implemented".format(type))

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader


    def _prepare_optimizer_and_scheduler(self, t_total, warmup_step_ratio, weight_decay, learning_rate,
                                         embedding_learning_rate, adam_epsilon):

        cur_model = self.model.module if hasattr(self.model, 'module') else self.model
        warmup_steps = int(t_total * warmup_step_ratio)
        no_decay = ['bias', 'LayerNorm.weight']

        print(cur_model.model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        ret_dict = {"optimizer": optimizer, "scheduler": scheduler}

        if self.config.method == "ptuning":

            embedding_parameters = [{'params': [p for p in cur_model.prompt_encoder.parameters()]}]
            embedding_optimizer = AdamW(embedding_parameters, lr=embedding_learning_rate, eps=adam_epsilon)
            embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

            ret_dict["embedding_optimizer"] = embedding_optimizer
            ret_dict["embedding_scheduler"] = embedding_scheduler

        return ret_dict



    def _train(self,
               train_data, dev32_data, unlabeled_data,
               per_gpu_train_batch_size, per_gpu_eval_batch_size, per_gpu_unlabeled_batch_size,
               learning_rate,embedding_learning_rate,weight_decay,adam_epsilon,warmup_step_ratio,
               gradient_accumulation_steps,max_grad_norm,
               train_priming, sampler_seed, eval_priming, priming_num,
               max_steps, num_train_epochs,
               metrics, decoding_strategy, alpha, temperature,
               early_stop_epoch, every_eval_step, pattern_iter_output_dir,
               n_gpu, device,
               **_):

        train_dataloader=self._prepare_dataloader("train", train_data, per_gpu_train_batch_size, train_priming,
                                                  sampler_seed, n_gpu, labelled=True)



        if self.config.method == "adapet":
            extra_dataloader = self._prepare_dataloader("extra", train_data, per_gpu_train_batch_size, train_priming,
                                                  sampler_seed, n_gpu, labelled=True)
            extra_iter = extra_dataloader.__iter__()
        elif self.config.method == "lm_training":
            extra_dataloader = self._prepare_dataloader("extra", unlabeled_data, per_gpu_unlabeled_batch_size,
                                                        train_priming,
                                                        sampler_seed, n_gpu, labelled=False)
            extra_iter = extra_dataloader.__iter__()
        else:
            extra_dataloader, extra_iter = None, None


        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        logger.info("\n")
        logger.info("num_steps_per_dataset:")
        logger.info(len(train_dataloader) // gradient_accumulation_steps)
        logger.info("total_steps:")
        logger.info(t_total)
        logger.info("num_train_epochs:")
        logger.info(num_train_epochs)

        ret_dict = self._prepare_optimizer_and_scheduler(t_total, warmup_step_ratio, weight_decay, learning_rate,
                                         embedding_learning_rate, adam_epsilon)
        optimizer, scheduler = ret_dict["optimizer"], ret_dict["scheduler"]
        embedding_optimizer, embedding_scheduler = None, None
        if self.config.method == "ptuning":
            embedding_optimizer, embedding_scheduler = ret_dict["embedding_optimizer"], ret_dict["embedding_scheduler"]

        writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "writer_logs"))


        """
        unlabeled_dataloader, unlabeled_iter = None, None
        ada_dataloader, ada_iter = None,None

        if use_adapet_loss:
            ada_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            ada_dataset = self._generate_dataset(task_train_data, labelled=True, priming=train_priming)
            ada_sampler = RandomSampler(ada_dataset, sampler_seed)
            ada_dataloader = DataLoader(ada_dataset, sampler=ada_sampler, batch_size=ada_batch_size)
            ada_iter = ada_dataloader.__iter__()


        if lm_training or use_logits:

            if unlabeled_data is not None:
                all_data = unlabeled_data + task_train_data
            else:
                all_data = task_train_data

            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            unlabeled_dataset = self._generate_dataset(all_data, labelled=False, priming=train_priming)
            unlabeled_sampler = RandomSampler(unlabeled_dataset, sampler_seed)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler, batch_size=unlabeled_batch_size)
            unlabeled_iter = unlabeled_dataloader.__iter__()


        assert use_logits == False
        if use_logits:
            train_dataloader = unlabeled_dataloader
        """


        logger.info("\n")
        logger.info("dev32_data performance before training.")
        dev32_scores = self.evaluate(dev32_data, per_gpu_eval_batch_size, n_gpu, device, metrics,
                                     decoding_strategy, eval_priming,
                                     priming_num,
                                     priming_data=train_data)["scores"]
        logger.info(dev32_scores)
        logger.info("\n")

        """
        logger.info("eval_data performance before training.")
        dev_scores = self.evaluate(eval_data, per_gpu_eval_batch_size, n_gpu, device, metrics,
                                     decoding_strategy, eval_priming,
                                     priming_num,
                                     priming_data=train_data)
        logger.info(dev_scores)
        """


        best_scores = {key:0.0 for key in metrics}
        cur_early_stop_epoch = 0
        best_global_step, best_tr_loss = 0, 0.0

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.zero_grad()
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(device) for k, t in batch.items()}


                extra_batch = None
                if extra_dataloader is not None:
                    while extra_batch is None:
                        try:
                            extra_batch = extra_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting adapet dataset")
                            extra_iter = extra_dataloader.__iter__()

                if self.config.method == "adapet":
                    lm_input_ids = extra_batch['input_ids']
                    lm_block_flags = extra_batch['block_flags']
                    if "labels" in extra_batch:
                        lm_labels = extra_batch["labels"]
                        lm_fake_labels = np.random.randint(len(self.config.label_list), size=len(lm_input_ids))
                        tgt = torch.from_numpy(lm_fake_labels).long() == lm_labels
                    else:
                        lm_fake_labels = None
                        tgt = None

                    extra_batch["original_input_ids"] = lm_input_ids
                    extra_batch["tgt"] = tgt
                    extra_batch['input_ids'], extra_batch['mlm_labels'] = self._mask_tokens(lm_input_ids, lm_block_flags, lm_fake_labels)
                    extra_batch = {k: t.to(device) for k, t in extra_batch.items()}

                elif self.config.method == "lm_training":
                    unlabeled_lm_input_ids = extra_batch['input_ids']
                    unlabeled_lm_block_flags = extra_batch["block_flags"]
                    extra_batch['input_ids'], extra_batch['mlm_labels'] = self._mask_tokens(unlabeled_lm_input_ids, unlabeled_lm_block_flags)
                    extra_batch = {k: t.to(device) for k, t in extra_batch.items()}


                train_step_inputs = {
                    'extra_batch': extra_batch,
                    'alpha': alpha,
                    'temperature': temperature,
                    "priming": train_priming,
                    "priming_num": priming_num,
                }

                loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None
                if loss is None:
                    loss = self.model.train_step(batch, **train_step_inputs)
                    # loss = TRAIN_STEP_FUNCTIONS[self.wrapper_type](self)(batch, **train_step_inputs)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    if self.config.method == "ptuning":
                        embedding_optimizer.step()
                        embedding_scheduler.step()

                    self.model.zero_grad()
                    global_step += 1


                    """
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        logs['loss'] = loss_scalar

                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar


                        if self.config.use_continuous_prompt:
                            embedding_learning_rate_scalar = embedding_scheduler.get_lr()[0]
                            logs["embedding_learning_rate"] = embedding_learning_rate_scalar
                            writer.add_scalar("embedding_learning_rate", embedding_learning_rate_scalar,
                                              global_step=global_step)

                        writer.add_scalar("train_loss", loss_scalar, global_step=global_step)
                        writer.add_scalar("learning_rate", learning_rate_scalar, global_step=global_step)
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{'step': global_step}}))
                    """


                    if every_eval_step > 0 and global_step % every_eval_step == 0:
                        dev32_scores = self.evaluate(dev32_data, per_gpu_eval_batch_size, n_gpu, device, metrics,
                                                     decoding_strategy, eval_priming, priming_num,
                                                     priming_data=train_data)["scores"]
                        print(dev32_scores)

                        save_checkpoint = True
                        for metric in metrics:
                            writer.add_scalar("dev32_" + metric, dev32_scores[metric], global_step=global_step)
                            if dev32_scores[metric] < best_scores[metric]:
                                save_checkpoint = False
                        if save_checkpoint:
                            if best_scores == dev32_scores:
                                cur_early_stop_epoch += 1
                            else:
                                cur_early_stop_epoch = 0

                            best_tr_loss = tr_loss
                            best_scores = dev32_scores
                            best_global_step = global_step

                            logger.info("\n")
                            logger.info("best_global_step: {} | saving models at {}...".format(best_global_step, pattern_iter_output_dir))
                            logger.info("best_scores:")
                            logger.info(best_scores)
                            self.save(pattern_iter_output_dir)
                        else:
                            cur_early_stop_epoch += 1
                        logger.info("early_stop_epoch: " + str(cur_early_stop_epoch))
                        logger.info("\n\n")

                if 0 < max_steps < global_step or cur_early_stop_epoch >= early_stop_epoch:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step or cur_early_stop_epoch >= early_stop_epoch:
                train_iterator.close()
                break
        # return global_step, (tr_loss / global_step if global_step > 0 else -1)
        return best_global_step, (best_tr_loss / best_global_step if best_global_step > 0 else -1)





    def _eval(self,
              eval_data,
              device,
              per_gpu_eval_batch_size,
              n_gpu,
              eval_priming,
              priming_num,
              decoding_strategy) -> Dict:

        eval_dataloader = self._prepare_dataloader("eval", eval_data, per_gpu_eval_batch_size, eval_priming,
                                                   None, n_gpu, labelled=True)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():
                # some tasks require special evaluation
                logits = self.task_helper.eval_step(
                    batch, decoding_strategy=decoding_strategy) if self.task_helper else None

                if logits is None:
                    logits = self.model.eval_step(batch)
                    # logits = EVALUATION_STEP_FUNCTIONS[self.wrapper_type](self)(batch, priming=eval_priming, #
                    # priming_num=priming_num)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }



    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            "block_flags": torch.tensor([f.block_flags for f in features], dtype=torch.long)
        }
        """
        if self.wrapper_type == PLM_WRAPPER:
            feature_dict['perm_mask'] = torch.tensor([f.perm_mask for f in features], dtype=torch.float)
            feature_dict['target_mapping'] = torch.tensor([f.target_mapping for f in features], dtype=torch.float)
        """
        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming)
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features



    def _mask_tokens(self, original_input_ids, block_flags, fake_labels=None):

        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        input_ids = original_input_ids.clone()
        if fake_labels is not None:
            for idx in range(len(input_ids)):
                lab_map = {i:lab for i, lab in enumerate(self.config.label_list)}
                lab = lab_map[fake_labels[idx]]
                fake_label_token = self.preprocessor.pvp.verbalize(lab)
                # print(fake_label_token)
                fake_label_id = get_verbalization_ids(fake_label_token, self.tokenizer, False)
                # print(fake_label_id)
                # print(input_ids[idx])
                input_ids[idx] = torch.where(input_ids[idx]==self.tokenizer.mask_token_id, torch.tensor(fake_label_id), input_ids[idx])
                # print(input_ids[idx])
                # print("\n")

        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)

        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        # print(input_ids)

        # TODO
        new_block_flags = (block_flags != -1).bool()
        # ((False in new_block_flags) == False)
        masked_indices = masked_indices & new_block_flags

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # print(input_ids)


        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # print(input_ids)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels
