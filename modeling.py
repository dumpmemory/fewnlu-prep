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
import ast
import json
import os
import random
import statistics
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
from methods.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from methods.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

logger = log.get_logger('root')





def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model




def ensemble_merge_logits(candidates, output_file, input_dir, reduction):
    all_logits_lists = []
    for subdir in candidates:

        results_file = os.path.join(input_dir, subdir, "p0-i0", 'results.txt')
        logits_file = os.path.join(input_dir, subdir, "p0-i0", "eval_logits.txt")
        logits = []
        if not os.path.exists(logits_file) or not os.path.exists(results_file):
            print(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        print("Score = {}, #Logits = {}, #Labels = {}".format(
            result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction="mean")
    merged_loglist.save(output_file)


def only_ensemble(candidates, ensemble_input_dir, eval_data, config, label_list, reduction, priming_data=None):

    output_file = os.path.join(ensemble_input_dir, "test_merged_best_logits.txt")
    ensemble_merge_logits(candidates, output_file, ensemble_input_dir, reduction)
    logits = LogitsList.load(output_file).logits
    logits = np.array(logits)

    predictions = np.argmax(logits, axis=1)
    scores = {}

    out_label_ids, question_ids = [], []
    label_map = {label: i for i, label in enumerate(label_list)}

    for input_example in eval_data:
        label = label_map[input_example.label] if input_example.label is not None else -100
        out_label_ids.append(label)
        if 'question_idx' in input_example.meta:
            question_ids.append(input_example.meta["question_idx"])

    out_label_ids = np.array(out_label_ids)
    question_ids = np.array(question_ids)

    results = {}
    metrics = config.metrics if config.metrics else ['acc']
    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, out_label_ids)
        elif metric == 'f1':
            scores[metric] = f1_score(out_label_ids, predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(out_label_ids, predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, out_label_ids, question_ids)
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    print(results)
    return results


def train_ipet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
               ensemble_eval_config: EvalConfig, ipet_config: IPetConfig, final_model_config: WrapperConfig,
               final_train_config: TrainConfig, final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str,
               ensemble_repetitions: int = 3, final_repetitions: int = 1, reduction: str = 'wmean',
               train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None, dev32_data: List[InputExample] = None,
               eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a new iPET model for a given data_utils.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param ipet_config: the iPET training configuration
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """
    for gen in range(ipet_config.generations):

        gen_output_dir = os.path.join(output_dir, f'g{gen}')

        # Step 1: Train an ensemble of models corresponding to individual patterns
        ipet_data_dir = os.path.join(output_dir, f'g{gen - 1}', 'next-gen-train-data') if gen > 0 else None

        train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                           gen_output_dir, ipet_data_dir=ipet_data_dir, dev32_data=dev32_data,
                           repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                           eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=True)

        # Step 2: Use the model to annotate examples for the next generation
        original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
        num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data))
        generate_ipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                 labels=ensemble_model_config.label_list, logits_dir=gen_output_dir,
                                 output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'), reduction=reduction,
                                 num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                 n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed)

    # Step 3: Merge the annotations created by each individual model
    logits_dir = os.path.join(output_dir, f'g{ipet_config.generations - 1}')
    logits_file = os.path.join(logits_dir, 'unlabeled_logits.txt')
    merge_logits(logits_dir, logits_file, reduction)
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 4: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data, dev32_data=dev32_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval)


def train_pet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
              ensemble_eval_config: EvalConfig, final_model_config: WrapperConfig, final_train_config: TrainConfig,
              final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str, ensemble_repetitions: int = 3,
              final_repetitions: int = 1, reduction: str = 'wmean', train_data: List[InputExample] = None, dev32_data: List[InputExample] = None,
              unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None, do_train: bool = True,
              do_eval: bool = True, no_distillation: bool = False, seed: int = 42):
    """
    Train and evaluate a new PET model for a given data_utils.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param no_distillation: if true, no distillation is performed
    :param seed: the random seed to use
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns
    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids, output_dir,
                       repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval, dev32_data=dev32_data,
                       save_unlabeled_logits=not no_distillation, seed=seed)

    if no_distillation:
        return

    # Step 2: Merge the annotations created by each individual model
    logits_file = os.path.join(output_dir, 'unlabeled_logits.txt')
    merge_logits(output_dir, logits_file, reduction)
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 3: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data, dev32_data=dev32_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval, seed=seed)


def train_classifier(model_config: WrapperConfig,
                     train_config: TrainConfig,
                     eval_config: EvalConfig,
                     output_dir: str,
                     repetitions: int = 3,
                     train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None,
                     eval_data: List[InputExample] = None,
                     dev32_data: List[InputExample] = None,
                     do_train: bool = True,
                     do_eval: bool = True,
                     seed: int = 42):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_pet_ensemble(model_config,
                       train_config,
                       eval_config,
                       pattern_ids = [0],
                       output_dir=output_dir,
                       repetitions=repetitions,
                       train_data=train_data,
                       unlabeled_data=unlabeled_data,
                       eval_data=eval_data,
                       dev32_data=dev32_data,
                       do_train=do_train,
                       do_eval=do_eval,
                       seed=seed)













def train_pet_ensemble(model_config: WrapperConfig,
                       train_config: TrainConfig,
                       eval_config: EvalConfig,
                       pattern_ids: List[int],
                       output_dir: str,
                       ipet_data_dir: str = None,
                       repetitions: int = 3,
                       train_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None,
                       dev32_data: List[InputExample] = None,
                       do_train: bool = True,
                       do_eval: bool = True,
                       save_unlabeled_logits: bool = False,
                       seed: int = 42):

    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:

        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            # TODO

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")


                if save_unlabeled_logits and not os.path.exists(os.path.join(pattern_iter_output_dir, 'logits.txt')):
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                    logits = evaluate(wrapper, unlabeled_data, eval_config)['logits']
                    save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

                if (not do_train) and do_eval:
                    pass
                else:
                    continue


            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            # Training
            if do_train:

                if ipet_data_dir:
                    print("loading ipet data.")
                    p = os.path.join(ipet_data_dir, 'p{}-i{}-train.bin'.format(pattern_id, iteration))
                    ipet_train_data = InputExample.load_examples(p)
                    for example in ipet_train_data:
                        example.logits = None
                else:
                    ipet_train_data = None

                results_dict.update(train_single_model(eval_data, dev32_data, pattern_iter_output_dir,
                                                       wrapper, train_data, train_config, eval_config,
                                                       ipet_train_data=ipet_train_data,
                                                       unlabeled_data=unlabeled_data))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                # logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                # wrapper.save(pattern_iter_output_dir)
                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if save_unlabeled_logits:
                    logits = evaluate(wrapper, unlabeled_data, eval_config)['logits']
                    save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info("Starting evaluation...")
                # if not wrapper:
                #    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
                logger.info("restoring checkpoint from {}".format(pattern_iter_output_dir))

                eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)
                dev32_eval_result = evaluate(wrapper, dev32_data, eval_config, priming_data=train_data)

                save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'dev32_eval_logits.txt'), dev32_eval_result['logits'])

                scores = eval_result['scores']
                logger.info("--- eval_data RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(scores)
                logger.info("--- dev32_data RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(dev32_eval_result["scores"])

                results_dict['test_set_after_training'] = scores
                results_dict["dev32_set_after_training"] = dev32_eval_result["scores"]

                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in scores.items():
                    results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def train_single_model(eval_data, dev32_data: List[InputExample], pattern_iter_output_dir,
                       model: TransformerModelWrapper,
                       train_data: List[InputExample],
                       config: TrainConfig,
                       eval_config: EvalConfig = None,
                       ipet_train_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None,
                       return_train_set_results: bool = True):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param ipet_train_data: an optional list of iPET training examples to use
    :param unlabeled_data: an optional list of unlabeled examples to use
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    results_dict = {}
    """
    if train_data and return_train_set_results:
        results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
    """

    if ipet_train_data is not None:
        train_data = train_data + ipet_train_data

    # TODO: train_priming
    all_train_data = []
    all_unlabeled_data = []

    priming_num = config.priming_num
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
    if not ipet_train_data:
        all_train_data += ipet_train_data
    """

    if not all_train_data and not config.use_logits:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(config.embedding_learning_rate, eval_data, dev32_data, eval_config,
                                           config.sampler_seed, config.every_eval_step, pattern_iter_output_dir,
            all_train_data, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            # unlabeled_data=unlabeled_data if config.lm_training or config.use_logits else None,
            unlabeled_data=all_unlabeled_data if config.lm_training or config.use_logits else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            train_priming=config.priming,
            eval_priming=eval_config.priming,
            priming_num=config.priming_num,
            use_adapet_loss=config.use_adapet_loss,
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # if train_data and return_train_set_results:
    #    best_model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    #    results_dict['train_set_after_training'] = evaluate(best_model, train_data, eval_config)['scores']['acc']

    return results_dict


def evaluate(model: TransformerModelWrapper,
             eval_data: List[InputExample],
             config: EvalConfig,
             priming_data: List[InputExample] = None,
             cali=0.5) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    all_eval_data = []
    if config.priming and priming_data:
        logger.info("priming when evaluting.")

        random.seed(42)
        for example in eval_data:
            """
            priming_example = random.sample(priming_data, k=config.priming_num)
            example.meta['priming_data'] = priming_example
            all_eval_data.append(example)

            """
            k = config.priming_num
            for idx in range(int(len(priming_data) / k)):
                priming_example = priming_data[idx * k: (idx + 1) * k]
                example.meta['priming_data'] = priming_example
                all_eval_data.append(example)

    else:
        logger.info("NOT priming when evaluting.")
        all_eval_data = eval_data



    metrics = config.metrics if config.metrics else ['acc']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(all_eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy, priming=config.priming,
                         priming_num=config.priming_num)

    if not config.use_cali:
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

    # TODO: cali
    p_cf = [1 - cali, cali]
    W = np.linalg.inv(np.identity(len(p_cf)) * p_cf)
    results2 = np.array(results['logits'])
    origin_probs = []
    for each_result in results2:
        origin_prob = softmax(each_result)
        origin_probs.append(origin_prob)
    cali_prob = np.matmul(W, np.transpose(origin_probs))
    cali_prob = np.transpose(cali_prob)
    predictions = np.argmax(cali_prob, axis=1)
    false_0 = 0
    false_1 = 0
    for i in range(0, len(predictions)):
        if results['labels'][i] == 1 and predictions[i] == 0:
            false_0 = false_0 + 1
        if results['labels'][i] == 0 and predictions[i] == 1:
            false_1 = false_1 + 1
    print("cali:")
    print(false_0, false_1)
    cali_prob = results['logits']
    # print(cali_prob)
    pred = []
    for i in range(0, len(cali_prob)):
        pred.append([softmax(results['logits'][i])[1], results['labels'][i]])
    pred = sorted(pred)
    origin_pred = np.argmax(cali_prob, axis=1)
    f_0 = 0
    f_1 = 0
    for i in range(0, len(origin_pred)):
        if results['labels'][i] == 1 and origin_pred[i] == 0:
            f_0 = f_0 + 1
        if results['labels'][i] == 0 and origin_pred[i] == 1:
            f_1 = f_1 + 1

    ac = 0
    max_ac = 0
    for i in range(0, len(results2)):
        if pred[i][1] == 1:
            ac = ac + 1
    max_ac = ac
    calis = 0
    for i in range(0, len(results2)):
        if pred[i][1] == 0:
            ac = ac + 1
        elif pred[i][1] == 1:
            ac = ac - 1
        if max_ac <= ac:
            max_ac = ac
            calis = pred[i][0]

    print("origin:")
    print(f_0, f_1)
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
    results['calis'] = calis
    return results






def merge_logits(logits_dir: str, output_file: str, reduction: str):
    """
    Merge the logits predicted for unlabeled examples by multiple models.

    :param logits_dir: a directory for which each sub-directory corresponds to a pretrained model and contains
           both a file ``results.txt`` containing that model's results on the training set and a file ``logits.txt``
           containing that model's predictions for the unlabeled data.
    :param output_file: the file to which the merged logits for all unlabeled examples are written.
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    """
    subdirs = next(os.walk(logits_dir))[1]
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=reduction)
    merged_loglist.save(output_file)


def merge_logits_lists(logits_lists: List[LogitsList], reduction: str = 'mean') -> LogitsList:
    """
    Merge a list of :class:`LogitsList` objects.

    :param logits_lists: the lists to merge
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :return: the merged list
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)



