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
This script can be used to train and evaluate either a few-shot method on
one of the supported tasks and datasets.
"""

import json
import shutil
import time
from collections import defaultdict
from typing import Dict
import statistics
import random
import itertools

from arguments import get_args
from configs import get_wrapper_config, get_train_eval_config, get_data_config
import os
import torch
import log
from utils import save_predictions, set_seed, save_logits, InputExample
from augmentation import generate_ipet_train_sets
from wrapper import TransformerModelWrapper
from tasks.dataloader import load_dataset, DATASETS

from global_vars import DEFAULT_METRICS, TRAIN_EVAL_CONFIG_NAME

logger = log.get_logger()

# def _write_results(path: str, results: Dict):
#     with open(path, 'w') as fh:
#         for metric in results.keys():
#             for pattern_id, values in results[metric].items():
#                 mean = statistics.mean(values)
#                 stdev = statistics.stdev(values) if len(values) > 1 else 0
#                 result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
#                 logger.info(result_str)
#                 fh.write(result_str + '\n')
#         for metric in results.keys():
#             all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
#             all_mean = statistics.mean(all_results)
#             all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
#             result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
#             logger.info(result_str)
#             fh.write(result_str + '\n')



def _write_results(path: str, results: Dict, dev32_results=None):

    ret_dict = {"dev32": {}, "dev": {}}
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
            ret_dict["dev"][metric] = all_mean
            # ret_dict["dev"]["all_stdev"][metric] = all_stdev

        if dev32_results is not None:
            for metric in dev32_results.keys():
                all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
                all_mean = statistics.mean(all_results)
                all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
                result_str = "{}-dev32-all-p: {} +- {}".format(metric, all_mean, all_stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')
                ret_dict["dev32"][metric] = all_mean
                # ret_dict["dev32"]["all_stdev"][metric] = all_stdev
    return ret_dict

def prepare_splited_data(args,train_data,dev32_data):
    train_datas=[];dev32_datas=[]
    if args.few_shot_setting in ['fix_setting','dev32_setting']:
        train_datas.append(train_data);dev32_datas.append(dev32_data)
    else:
        all_data=train_data+dev32_data
        cur_all_data = {}
        # logger.info(args.task_name)
        if args.task_name not in ["multirc", 'copa']:
            assert len(train_data) == len(dev32_data)
            number = len(train_data)
        else:
            cur_all_data = {}
            for item in all_data:
                guid  = "-".join(item.guid.split("-")[:-1])
                if guid in cur_all_data:
                    cur_all_data[guid].append(item)
                else:
                    cur_all_data[guid] = [item]
            all_num = len(cur_all_data)
            number = int(all_num / 2)
            cur_all_data = list(cur_all_data.items())
        K=args.cv_k
        for idx in range(K):
            if args.task_name not in ['multirc','copa']:
                random.shuffle(all_data)
                cur_train_data = all_data[:number]
                cur_dev32_data = all_data[number:]
            else:
                random.shuffle(cur_all_data)
                cur_train_data_set = [data_list for (guid, data_list) in cur_all_data[:number]]
                cur_dev32_data_set = [data_list for (guid, data_list) in cur_all_data[number:]]
                cur_train_data = list(itertools.chain.from_iterable(cur_train_data_set))
                cur_dev32_data = list(itertools.chain.from_iterable(cur_dev32_data_set))
            train_datas.append(cur_train_data)
            dev32_datas.append(cur_dev32_data)
    return train_datas,dev32_datas 

def iterative_run(train_datas, dev32_datas, eval_data, wrapper_config, train_eval_config, unlabeled_data=None, aug_data=None, output_dir=None):
    output_dir = output_dir if output_dir is not None else wrapper_config.output_dir
    if train_eval_config.generations==1:
        results=run(train_datas, dev32_datas, eval_data, wrapper_config, train_eval_config, output_dir, unlabeled_data, aug_data, save_unlabeled_logits=False)
        return results

    for gen in range(train_eval_config.generations):
        gen_output_dir = os.path.join(output_dir, f'g{gen}')
        if gen>0:
            ipet_data_dirs={}
            if unlabeled_data is not None:
                ipet_data_dirs['unlabeled']=os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data')
            if aug_data is not None and train_eval_config.relabel_aug_data==True:
                ipet_data_dirs['aug']=os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data')
        else:
            ipet_data_dirs=None
        if wrapper_config.arch_method=='noisy_student':
            train_eval_config.use_dropout=True if gen>0 else False
        # ipet_data_dirs = os.path.join(output_dir, f'g{gen - 1}','next-gen-train-data') if gen > 0 else None
        results=run(train_datas, dev32_datas, eval_data, wrapper_config, train_eval_config, gen_output_dir, unlabeled_data, aug_data, ipet_data_dirs, save_unlabeled_logits=True)

        if wrapper_config.arch_method in ['ipet', 'noisy_student']:
            assert (unlabeled_data is not None) or (aug_data is not None)
            logger.info("Augmenting data by self-labeling unlabeled data.")
            original_data_size = len(train_datas[0]) if train_datas else 10 / train_eval_config.ipet_scale_factor
            num_new_examples = int(original_data_size * (train_eval_config.ipet_scale_factor ** (gen + 1)) - len(train_datas[0]))
            if unlabeled_data is not None:
                generate_ipet_train_sets(train_datas=train_datas,
                                        unlabeled_data=unlabeled_data,
                                        labels=wrapper_config.label_list,
                                        logits_dir=gen_output_dir,
                                        output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'),
                                        reduction="mean",
                                        num_new_examples=num_new_examples,
                                        logits_percentage=train_eval_config.ipet_logits_percentage,
                                        n_most_likely=train_eval_config.ipet_n_most_likely if gen == 0 else -1,
                                        seed=train_eval_config.seed,
                                        logits_prefix='unlabeled',
                                        use_brother_fold_logits=train_eval_config.use_brother_fold_logits)
            if aug_data is not None and train_eval_config.relabel_aug_data==True:
                generate_ipet_train_sets(train_datas=train_datas,
                                     unlabeled_data=aug_data,
                                     labels=wrapper_config.label_list,
                                     logits_dir=gen_output_dir,
                                     output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'),
                                     reduction="mean",
                                     num_new_examples=num_new_examples,
                                     logits_percentage=train_eval_config.ipet_logits_percentage,
                                     n_most_likely=train_eval_config.ipet_n_most_likely if gen == 0 else -1,
                                     seed=train_eval_config.seed,
                                     logits_prefix='aug',
                                     use_brother_fold_logits=train_eval_config.use_brother_fold_logits)
        elif wrapper_config.method == "flipda":
            raise NotImplementedError("FlipDA to be implemented.")
    return results


def run(train_datas, dev32_datas, eval_data, wrapper_config, train_eval_config, output_dir=None, unlabeled_data=None, aug_data=None, ipet_data_dirs=None, save_unlabeled_logits=False):

    pattern_ids = train_eval_config.pattern_ids
    repetitions = train_eval_config.repetitions
    folds = train_eval_config.cv_k
    seed = train_eval_config.seed
    do_train = train_eval_config.do_train
    do_eval = train_eval_config.do_eval
    if output_dir is None:
        output_dir = wrapper_config.output_dir

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))

    set_seed(seed)
    assert len(train_eval_config.sampler_seeds) >= repetitions
    for pattern_id in pattern_ids:
        for fold in range(folds):
            train_data=train_datas[fold];dev32_data=dev32_datas[fold]
            for iteration in range(repetitions):
                results_dict = {}
                pattern_iter_output_dir = "{}/p{}/f{}-i{}".format(output_dir, pattern_id, fold, iteration)
                train_eval_config.sampler_seed = train_eval_config.sampler_seeds[iteration]

                if os.path.exists(pattern_iter_output_dir):
                    logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                    continue
                else:
                    os.makedirs(pattern_iter_output_dir)
                wrapper = TransformerModelWrapper(wrapper_config, pattern_id)
                if do_train:
                    ipet_train_data=None
                    if ipet_data_dirs is not None:
                        for (prefix,ipet_data_dir) in ipet_data_dirs.items():
                            p = os.path.join(ipet_data_dir, 'p{}-f{}-i{}-{}-train.bin'.format(pattern_id, fold, iteration, prefix))
                            tmp_ipet_train_data = InputExample.load_examples(p)
                            for example in tmp_ipet_train_data:
                                example.logits = None
                            if ipet_train_data is None:
                                ipet_train_data=tmp_ipet_train_data
                            else:
                                ipet_train_data+=tmp_ipet_train_data
                    if aug_data is not None and train_eval_config.relabel_aug_data==False:
                        ipet_train_data = ipet_train_data + aug_data

                    cur_results = wrapper.train(train_data, dev32_data, pattern_iter_output_dir, train_eval_config,
                                                unlabeled_data=unlabeled_data, ipet_train_data=ipet_train_data)
                    results_dict.update(cur_results)

                    with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                        fh.write(str(results_dict))

                    if train_eval_config.few_shot_setting == "fix_setting":
                        logger.info("Saving trained model at {} for fix-setting.".format(pattern_iter_output_dir))
                        wrapper.save(pattern_iter_output_dir)

                    train_eval_config.save(os.path.join(pattern_iter_output_dir, TRAIN_EVAL_CONFIG_NAME))
                    logger.info("Saving complete")

                    if save_unlabeled_logits:
                        if unlabeled_data is not None:
                            unlabeled_logits = wrapper.evaluate(unlabeled_data,
                                train_eval_config.per_gpu_eval_batch_size,
                                train_eval_config.n_gpu,
                                train_eval_config.device,
                                train_eval_config.metrics,
                                train_eval_config.decoding_strategy,
                                train_eval_config.eval_priming,
                                train_eval_config.priming_num, priming_data=train_data)['logits']
                            save_logits(os.path.join(pattern_iter_output_dir, 'unlabeled_logits.txt'), unlabeled_logits)
                            logger.info("unlabeled logits saved.")

                        if aug_data is not None and train_eval_config.relabel_aug_data==True:
                            aug_logits = wrapper.evaluate(aug_data,
                                train_eval_config.per_gpu_eval_batch_size,
                                train_eval_config.n_gpu,
                                train_eval_config.device,
                                train_eval_config.metrics,
                                train_eval_config.decoding_strategy,
                                train_eval_config.eval_priming,
                                train_eval_config.priming_num, priming_data=train_data)['logits']
                            save_logits(os.path.join(pattern_iter_output_dir, 'aug_logits.txt'), aug_logits)
                            logger.info("augmented logits saved.")

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

                    dev32_eval_result = wrapper.evaluate(dev32_data,
                        train_eval_config.per_gpu_eval_batch_size,
                        train_eval_config.n_gpu,
                        train_eval_config.device,
                        train_eval_config.metrics,
                        train_eval_config.decoding_strategy,
                        train_eval_config.eval_priming,
                        train_eval_config.priming_num, priming_data=train_data)

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

                    for metric, value in dev32_eval_result['scores'].items():
                        dev32_results[metric][pattern_id].append(value)

                    wrapper.model.cpu()
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        final_results=_write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
        return final_results
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def process_args(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and args.overwrite_output_dir:
        shutil.rmtree(args.output_dir)
        # pass

    if args.method == "sequence_classfier":
        args.use_cloze=False
    elif args.method in ["pet", "adapet", "lm_training"]:
        args.use_cloze=True
        args.use_continuous_prompt=False
    elif args.method == "ptuning":
        args.use_cloze=True
        args.use_continuous_prompt=True

    if args.arch_method=='default': #TODO
        args.generations=1

    if args.few_shot_setting in ['fix_setting','dev32_setting']:
        args.cv_k=1

    ### Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.task_name.lower()
    ### get metrics
    metrics = DATASETS[args.dataset_name]["metrics"]
    args.metrics = metrics.get(args.task_name, DEFAULT_METRICS)
    return args

def main():
    start_time = time.time()
    args = get_args()
    set_seed(args.seed)
    args = process_args(args)
    processors = DATASETS[args.dataset_name]["processors"]
    if args.task_name not in processors:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = processors[args.task_name](args.task_name)
    args.label_list = processor.get_labels()

    logger.info("\n")
    logger.info("Parameters: {}".format(args))
    logger.info("\n")

    ### prepare configurations
    data_config = get_data_config(args)
    wrapper_config = get_wrapper_config(args)
    train_eval_config = get_train_eval_config(args)

    ### prepare data
    train_data, dev32_data, eval_data, unlabeled_data = load_dataset(data_config)
    if args.aug_data_dir is not None:
        aug_data = processor._create_examples(args.aug_data_dir,"aug")
    else:
        aug_data = None

    logger.info('train_data: {}, dev32_data: {}, eval_data: {}'.format(len(train_data),len(dev32_data),len(eval_data)))

    start_time = time.time()
    # prepare train_data and dev32_data
    train_datas,dev32_datas=prepare_splited_data(args,train_data,dev32_data)
    results=iterative_run(train_datas, dev32_datas, eval_data, wrapper_config, train_eval_config, unlabeled_data, aug_data, output_dir=args.output_dir)
        #     for metric in args.metrics:
        #         all_results["dev32"][metric]+=results["dev32"][metric]
        #         all_results["dev"][metric]+=results["dev"][metric]
        # results={x:{y1:y2/K for (y1,y2) in y.items()} for (x,y) in all_results.items()}
    end_time=time.time()
    time_eclapsed = int(end_time-start_time)
    # hyper_params = args.output_dir.split("/")[-1]
    template_name=['task_name','few_shot_setting','max_steps','warmup_ratio','gradient_accumulation_steps','per_gpu_train_batch_size','lr','pattern','max_seq_length','every_eval_ratios']
    template_values=[args.task_name,args.few_shot_setting,args.max_steps,args.warmup_step_ratio,args.gradient_accumulation_steps,args.per_gpu_train_batch_size,args.learning_rate,args.pattern_ids,args.max_seq_length,args.every_eval_ratio]
    if args.method=='ptuning':
        template_name.append('embedding_learning_rate')
        template_values.append(args.embedding_learning_rate)
    hyper_params=(': {}, '.join(template_name)+': {},').format(*template_values)
    result_file_name = "factor_" + args.arch_method +"_"+ args.task_name + "_" + args.method + "_" +args.model_type + ".txt"
    with open(os.path.join("final_results", result_file_name), "a+") as f:
        if args.task_name in ["boolq", "rte", "wic", "wsc", "copa"]:
            f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t"  + str(time_eclapsed) +'\n')
        elif args.task_name == "cb":
            f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev32"]['f1-macro'])
                    + "\t" + str(results["dev"]['f1-macro'])
                    + "\t"  + str(time_eclapsed) + '\n')
        elif args.task_name == "multirc":
            f.write(hyper_params
                    + "\t" + str(results["dev32"]['acc'])
                    + "\t" + str(results["dev"]['acc'])
                    + "\t" + str(results["dev32"]['f1'])
                    + "\t" + str(results["dev"]['f1'])
                    + "\t" + str(results["dev32"]['em'])
                    + "\t" + str(results["dev"]['em'])
                    + "\t"  + str(time_eclapsed) + '\n')
    logger.info("\n")
    logger.info("Time elapsed: " + str(time_eclapsed))
    logger.info("\n")

if __name__ == "__main__":
    main()