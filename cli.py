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
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import copy
import json
import shutil
import time
from collections import defaultdict
import random
from typing import Tuple, Dict, List
import statistics

import numpy as np

from arguments import get_args
from configs import get_wrapper_config, get_train_eval_config
import os
import torch
import log
from methods.utils import save_predictions, set_seed, save_logits, InputExample, softmax, LogitsList, eq_div
from methods.wrapper import TransformerModelWrapper
from tasks.dataloader import load_dataset, DATASETS

logger = log.get_logger('root')

DEFAULT_METRICS = ["acc"]

def _write_results(path: str, results: Dict):
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')


def generate_ipet_train_sets(train_data: List[InputExample],
                             unlabeled_data: List[InputExample],
                             labels: List[str],
                             logits_dir: str,
                             output_dir: str,
                             reduction: str,
                             num_new_examples: int,
                             logits_percentage: float,
                             n_most_likely: int,
                             seed: int):
    subdirs = next(os.walk(logits_dir))[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))
    if train_data:
        train_examples_per_label = [sum(1 for ex in train_data if ex.label == label) for label in labels]
        multiplier = num_new_examples / len(train_data)
        examples_per_label = [int(epl * multiplier) for epl in train_examples_per_label]
        logger.info(f"Example distribution in the original dataset: {train_examples_per_label}")
    else:
        examples_per_label = eq_div(num_new_examples, len(labels))

    logger.info(f"Target distribution for the new dataset: {examples_per_label}")

    for example in unlabeled_data:
        example.label, example.logits = None, None

    logits_lists = {}

    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []
        # if not os.path.exists(results_file) or not os.path.exists(logits_file):
        if not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        """
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']
        """
        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        logits_lists[subdir] = loglist

    print(subdirs)
    for subdir in subdirs:

        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_ipet_train_set(
            other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
            logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
            rng_np=rng_np
        )
        InputExample.save_examples(subdir_train_set,
                                   os.path.join(output_dir, subdir + '-train.bin'))


def generate_ipet_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                            examples_per_label: List[int], logits_percentage: float, reduction: str = 'mean',
                            n_most_likely: int = -1, rng=None, rng_np=None) -> List[InputExample]:

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

    if not rng:
        rng = random.Random()
    if not rng_np:
        rng_np = np.random.RandomState()

    num_logits_lists = max(round(len(logits_lists) * logits_percentage), 1)
    logits_lists = rng.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
        logits = softmax(logits, axis=1).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
        logits = softmax(logits, axis=1).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [copy.deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples, num_examples=examples_per_label[idx], rng=rng_np)
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int, rng) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return rng.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()


def iterative_run(train_data, dev32_data, eval_data, unlabeled_data, wrapper_config, train_eval_config):
    output_dir = wrapper_config.output_dir
    for gen in range(train_eval_config.generations):
        gen_output_dir = os.path.join(output_dir, f'g{gen}')
        aug_data_dir = os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data') if gen > 0 else None
        run(train_data, dev32_data, eval_data, wrapper_config, train_eval_config,
            unlabeled_data, gen_output_dir, aug_data_dir)

        original_data_size = len(train_data) if train_data else 10 / train_eval_config.scale_factor
        num_new_examples = int(original_data_size * (train_eval_config.scale_factor ** (gen + 1)) - len(train_data))
        generate_ipet_train_sets(train_data=train_data,
                                 unlabeled_data=unlabeled_data,
                                 labels=wrapper_config.label_list,
                                 logits_dir=gen_output_dir,
                                 output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'),
                                 reduction="mean",
                                 num_new_examples=num_new_examples,
                                 logits_percentage=train_eval_config.logits_percentage,
                                 n_most_likely=train_eval_config.n_most_likely if gen == 0 else -1,
                                 seed=train_eval_config.seed)



def run(train_data, dev32_data, eval_data, wrapper_config, train_eval_config,
        unlabeled_data=None, output_dir=None, aug_data_dir=None):

    seed = train_eval_config.seed
    pattern_ids = train_eval_config.pattern_ids
    repetitions = train_eval_config.repetitions
    do_train = train_eval_config.do_train
    do_eval = train_eval_config.do_eval

    output_dir = output_dir if output_dir is not None else wrapper_config.output_dir
    results = defaultdict(lambda: defaultdict(list))

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):
            set_seed(seed)

            results_dict = {}
            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)
            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue
            else:
                os.makedirs(pattern_iter_output_dir)

            wrapper = TransformerModelWrapper(wrapper_config, pattern_id)

            if do_train:
                if aug_data_dir is not None:
                    p = os.path.join(aug_data_dir, 'p{}-i{}-train.bin'.format(pattern_id, iteration))
                    ipet_train_data = InputExample.load_examples(p)
                    for example in ipet_train_data:
                        example.logits = None
                else:
                    ipet_train_data = None

                cur_results = wrapper.train(train_data, dev32_data, pattern_iter_output_dir, train_eval_config,
                                            unlabeled_data=unlabeled_data, ipet_train_data=ipet_train_data)

                # train_single_model(eval_data, dev32_data, pattern_iter_output_dir,
                #                                       wrapper, train_data, train_config, eval_config,
                #                                       ipet_train_data=ipet_train_data,
                #                                       unlabeled_data=unlabeled_data)

                results_dict.update(cur_results)

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                train_eval_config.save(os.path.join(pattern_iter_output_dir, 'train_eval_config.json'))
                # eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if wrapper_config.method in ["ipet", 'noisy_student']:
                    logits = wrapper.evaluate(unlabeled_data,
                        train_eval_config.per_gpu_eval_batch_size,
                        train_eval_config.n_gpu,
                        train_eval_config.device,
                        train_eval_config.metrics,
                        train_eval_config.decoding_strategy,
                        train_eval_config.eval_priming,
                        train_eval_config.priming_num, priming_data=train_data)['logits']
                    save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)
                    logger.info("logits saved.")



                wrapper.model.cpu()
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

            if do_eval:
                logger.info("Starting evaluation...")
                logger.info("restoring checkpoint from {}".format(pattern_iter_output_dir))
                wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                eval_result = wrapper.evaluate(
                    eval_data,
                    train_eval_config.per_gpu_eval_batch_size,
                    train_eval_config.n_gpu,
                    train_eval_config.device,
                    train_eval_config.metrics,
                    train_eval_config.decoding_strategy,
                    train_eval_config.eval_priming,
                    train_eval_config.priming_num, priming_data=train_data)

                # = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)
                dev32_eval_result = wrapper.evaluate(dev32_data,
                    train_eval_config.per_gpu_eval_batch_size,
                    train_eval_config.n_gpu,
                    train_eval_config.device,
                    train_eval_config.metrics,
                    train_eval_config.decoding_strategy,
                    train_eval_config.eval_priming,
                    train_eval_config.priming_num, priming_data=train_data)

                # evaluate(wrapper, dev32_data, eval_config, priming_data=train_data)
                print(eval_result)
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

                wrapper.model.cpu()
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")




def main():

    start_time = time.time()

    ### get arguments
    args = get_args()

    """
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not \
            args.overwrite_output_dir:
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        shutil.rmtree(args.output_dir)
    """
    ### Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    ### get label list
    args.task_name = args.task_name.lower()
    PROCESSORS = DATASETS[args.dataset_name]["processors"]
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name](args.task_name)
    args.label_list = processor.get_labels()

    ### get metrics
    METRICS = DATASETS[args.dataset_name]["metrics"]
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    ### prepare data
    train_data, eval_data, dev32_data, unlabeled_data = load_dataset(args)



    ### prepare configurations
    wrapper_config = get_wrapper_config(args)
    train_eval_config = get_train_eval_config(args)

    # non-self-training methods
    if args.method in ["pet", "sequence_classifier", "ptuning"]:
        run(train_data, dev32_data, eval_data, wrapper_config, train_eval_config)

    elif args.method in ["adapet", "lm_training"]:
        run(train_data, dev32_data, eval_data, wrapper_config, train_eval_config, unlabeled_data)

    elif args.method in ["ipet","noisy_student"]:
        iterative_run(train_data, dev32_data, eval_data, unlabeled_data, wrapper_config, train_eval_config)

    else:
        raise NotImplementedError(f"Training method '{args.method}' not implemented.")

    logger.info("\n")
    logger.info("elapsed time: " + str(time.time() - start_time))
    logger.info("\n")


if __name__ == "__main__":
    main()