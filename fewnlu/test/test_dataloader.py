import os
from argparse import Namespace
from tasks.dataloader import load_dataset

if __name__ == '__main__':

    # test dataset
    args = Namespace()
    args.use_cloze = True
    args.task_name = "wsc"
    args.dataset_name = "superglue"
    data_dir = "/workspace/yanan/few-shot/FewGLUE_dev32/"
    args.seed = 42

    args.train_examples=-1
    args.dev_examples=-1
    args.dev32_examples=-1
    args.test_examples=-1
    args.split_examples_evenly=True
    args.eval_set = "dev"

    args.method = "models"

    for task_name in ["BoolQ", "RTE", "MultiRC", "WiC", "COPA", "ReCoRD", "CB"]:
        args.data_dir = os.path.join(data_dir, task_name)
        args.task_name = task_name.lower()
        train_data, dev32_data, eval_data, unlabeled_data = load_dataset(args)
        print("\n")

    args.data_dir = os.path.join(data_dir, "WSC")
    args.task_name = "wsc"
    args.use_cloze = True
    train_data, dev32_data, eval_data, unlabeled_data = load_dataset(args)
    print("\n")
    args.use_cloze = False
    train_data, dev32_data, eval_data, unlabeled_data = load_dataset(args)