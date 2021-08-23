import os
from argparse import Namespace

from transformers import AlbertTokenizer

from tasks.dataloader import load_dataset, DATASETS


if __name__ == '__main__':
    args = Namespace()

    args.use_cloze = True
    args.task_name = "rte"
    args.dataset_name = "superglue"
    data_dir = "/workspace/yanan/few-shot/FewGLUE_dev32/"
    args.seed = 42

    args.method = "models"

    tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")

    args.pattern_id = 1
    args.use_continuous_prompt = True
    args.max_seq_length = 256

    args.train_examples=-1
    args.dev_examples=-1
    args.dev32_examples=-1
    args.test_examples=-1
    args.split_examples_evenly=False
    args.eval_set = "dev"

    args.priming=True
    args.priming_num=1

    """
    for task_name in ["RTE"]:
        args.task_name = task_name.lower()
        pvps = DATASETS[args.dataset_name]["pvps"]
        pvp = pvps[args.task_name](tokenizer, args.pattern_id, args.use_cloze, args.use_continuous_prompt, args.seed, args.max_seq_length)

        example = InputExample(guid=None,
                               text_a="I'm crazy about Nintendo products.",
                               text_b="Oh so do I.",
                               label="True")

        outputs = pvp.encode(example, args.priming)
        print(outputs.input_ids)
        print(outputs.token_type_ids)
        print(outputs.block_flags)
        print(outputs.mlm_labels)
        print("\n")
    """

    for task_name in ["BoolQ", "RTE", "MultiRC", "WiC", "COPA", "ReCoRD", "CB", "WSC"]:
        args.data_dir = os.path.join(data_dir, task_name)
        args.task_name = task_name.lower()
        train_data, dev32_data, eval_data, unlabeled_data = load_dataset(args)
        print("\n")

        pvps = DATASETS[args.dataset_name]["pvps"]
        pvp = pvps[args.task_name](tokenizer, args.pattern_id, args.use_cloze, args.use_continuous_prompt, args.seed,
                                   args.max_seq_length)

        example = train_data[0]
        if args.priming:
            example.meta["priming_data"] = train_data[:1]

        outputs = pvp.encode(example, args.priming)
        print(outputs.guid)
        print(outputs.input_ids)
        print(outputs.token_type_ids)
        print(outputs.block_flags.count(-1) == args.pattern_id * (args.priming + 1))
        # print(outputs.mlm_labels)
        print(outputs.mlm_labels.index(1) == outputs.input_ids.index(4))
        print(outputs.mlm_labels.count(1) == outputs.input_ids.count(4))
        # print(outputs.label)
        # print(outputs.idx)
        # print(outputs.logits)
        print("\n")



