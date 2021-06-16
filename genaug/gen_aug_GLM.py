import os
import torch
from utils import get_sample_writer, get_log_dir, print_and_save_args
from configure_data import prepare_tokenizer
from tasks.seq2seq.finetune import metrics_func_provider
from tasks.seq2seq.dataset import Seq2SeqDataset, BlankLMDataset
from train_utils import setup_model_and_optimizer, train_step
from utils import print_rank_0
from utils import Timers
from utils import load_checkpoint, save_checkpoint

from model import GLMModel
from model import PyTorchDistributedDataParallel as TorchDDP
from model import DistributedDataParallel as LocalDDP
from fp16 import FP16_Module

import random
from random import shuffle
from methods.wrapper import MODEL_CLASSES
from methods import pvp

from pretrain_glm import initialize_distributed, set_random_seed
from arguments import get_args


def init_GLM_args():
	args = get_args()
	args_dict={'block_lm':True,'num_layers': 24, 'hidden_size': 1024, 'num_attention_heads': 16, \
	'max_position_embeddings': 512, 'tokenizer_model_type': 'bert-large-uncased', \
	'tokenizer_type': 'BertWordPieceTokenizer', 'load_pretrained': '/data/zhoujing/NLP/data/checkpoints/GLM/blocklm-large-blank', \
	'blank_maskratio': 0.3, \
	'src_seq_length': 512, 'tgt_seq_length': 100, 'min_tgt_length': 0, \
	'length_penalty': 1, 'no_repeat_ngram_size': 1, 'eval_batch_size': 8, \
	'finetune': True, 'task': 'blank', 'do_sample': True, 'num_return_sequences': 1, 'select_topk':True, 'temperature': 0.9,'num_beams': 10, \
	'nproc_per_node': 1, 'nnodes': 1, 'node_rank': 0, 'master_addr': 'localhost', 'master_port': 'shuf -n 1 -i 10000-65535'}
	for (x,y) in args_dict.items():
		setattr(args,x,y)
	return args


def init_model(args):
	torch.backends.cudnn.enabled=False
	initialize_distributed(args)
	set_random_seed(args.seed)
	tokenizer=prepare_tokenizer(args)
	model, optimizer, lr_scheduler=setup_model_and_optimizer(args, **{})
	model = model.module
	args.load=args.load_pretrained
	load_checkpoint(model,optimizer,lr_scheduler,args)
	return model


def test(args,model_kwargs={},test_lines=None, model=None):
	tokenizer=prepare_tokenizer(args)
	if model is None:
		model, optimizer, lr_scheduler=setup_model_and_optimizer(args,**model_kwargs)
		if args.load_pretrained is not None and not args.pretrained_bert and not args.load:
			module = model
			if isinstance(module, (LocalDDP, TorchDDP)):
				module = module.module
			if isinstance(module, FP16_Module):
				module = module.module
			if not isinstance(module, GLMModel):
				module = module.model
			args.load = args.load_pretrained
			load_checkpoint(module, optimizer, lr_scheduler, args)
			args.load = None
			# This is critical when only model is loaded. We should make sure
			# master parameters are also updated.
			if args.fp16:
				optimizer._model_params_to_master_params()
		if args.load is not None:
			load_checkpoint(model, optimizer, lr_scheduler, args)
			# This is critical when only model is loaded. We should make sure
			# master parameters are also updated.
			if args.fp16:
				optimizer._model_params_to_master_params()

	args.iteration = 0
	summary_writer = None
	if torch.distributed.get_rank() == 0:
		args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
		if os.path.exists(os.path.join(args.log_dir, "test_results.json")) and args.load is None and not args.overwrite:
			raise ValueError("Output directory ({}) already exists and is not empty.".format(args.log_dir))
		summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
		print_and_save_args(args, verbose=False, log_dir=args.log_dir)

	# Print setup timing.
	# print_rank_0('done with setups ...')
	# timers.log(['train/valid/test dataset/dataloder', 'callback function',
	# 			'model and optimizer', 'pretrained checkpoint'])
	# print_rank_0('training ...')
	datasets=[BlankLMDataset(args, 'dev', tokenizer, test_lines)]
	end_of_train_callback=metrics_func_provider(args,tokenizer,is_test=True,datasets=datasets)
	socre_dict,total_predictions,pred_blanks=end_of_train_callback(model, epoch=-1,output_predictions=True,return_predictions=True)
	print(total_predictions)
	print_rank_0('done :-)')
	return total_predictions, pred_blanks





from pretrain_glm import initialize_distributed,set_random_seed
from arguments import get_args

def main():
	torch.backends.cudnn.enabled = False
	args = get_args()
	args_dict={'block_lm':True,'num_layers': 24, 'hidden_size': 1024, 'num_attention_heads': 16, \
	'max_position_embeddings': 512, 'tokenizer_model_type': 'bert-large-uncased', \
	'tokenizer_type': 'BertWordPieceTokenizer', 'load_pretrained': '/data/zhoujing/NLP/data/checkpoints/GLM/blocklm-large-blank', \
	'blank_maskratio': 0.3, \
	'src_seq_length': 512, 'tgt_seq_length': 100, 'min_tgt_length': 0, \
	'length_penalty': 1, 'no_repeat_ngram_size': 1, 'eval_batch_size': 8, \
	'finetune': True, 'task': 'blank', 'do_sample': True, 'num_return_sequences': 1, 'select_topk':True, 'temperature': 0.9,'num_beams': 10}

	for (x,y) in args_dict.items():
		setattr(args,x,y)


	args.do_sample=False
	initialize_distributed(args)
	set_random_seed(args.seed)

	tokenizer=prepare_tokenizer(args)
	model, optimizer, lr_scheduler=setup_model_and_optimizer(args,**{})
	model=model.module
	args.load=args.load_pretrained
	load_checkpoint(model, optimizer, lr_scheduler, args)

	import os
	from methods import pvp
	from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
	from methods.wrapper import MODEL_CLASSES
	NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
	task_name='boolq'
	data_dir='/data/zhoujing/NLP/data/FewGLUE'
	train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]),TRAIN_SET,num_examples=-1)
	gen_func=get_generate_data(task_name,train_examples=train_examples,priming_model='GLM',version=2)
	source_texts,tgt_texts=gen_func(priming=True)

	import pdb
	pdb.set_trace()
	test(args,{},(source_texts[0]['No'],tgt_texts[0]['No']),model=model)

if __name__ == '__main__':
	main()








'''
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch $DISTRIBUTED_ARGS zj_finetune_glm.py

'''

'''
import os
import torch
from utils import get_sample_writer, get_log_dir, print_and_save_args
from configure_data import prepare_tokenizer
from tasks.seq2seq.finetune import metrics_func_provider
from tasks.seq2seq.dataset import Seq2SeqDataset, BlankLMDataset
from train_utils import setup_model_and_optimizer, train_step
from utils import print_rank_0
from utils import Timers
from utils import load_checkpoint, save_checkpoint

from model import GLMModel
from model import PyTorchDistributedDataParallel as TorchDDP
from model import DistributedDataParallel as LocalDDP
from fp16 import FP16_Module

from pretrain_glm import initialize_distributed,set_random_seed
from arguments import get_args
torch.backends.cudnn.enabled = False
args = get_args()
args_dict={'block_lm':True,'num_layers': 24, 'hidden_size': 1024, 'num_attention_heads': 16, \
'max_position_embeddings': 512, 'tokenizer_model_type': 'bert-large-uncased', \
'tokenizer_type': 'BertWordPieceTokenizer', 'load_pretrained': '/data/zhoujing/NLP/data/checkpoints/GLM/blocklm-large-blank', \
'blank_maskratio': 0.3, \
'src_seq_length': 512, 'tgt_seq_length': 100, 'min_tgt_length': 0, \
'length_penalty': 1, 'no_repeat_ngram_size': 1, 'eval_batch_size': 8, \
'finetune': True, 'task': 'blank', 'do_sample': True, 'num_return_sequences': 1, 'select_topk':True, 'temperature': 0.9,'num_beams': 10, \
'nproc_per_node': 1, 'nnodes': 1, 'node_rank': 0, 'master_addr': 'localhost', 'master_port': 'shuf -n 1 -i 10000-65535'}

for (x,y) in args_dict.items():
	setattr(args,x,y)


args.do_sample=False
initialize_distributed(args)
set_random_seed(args.seed)
import zj_finetune_glm
import imp
imp.reload(zj_finetune_glm)
tokenizer=prepare_tokenizer(args)
model, optimizer, lr_scheduler=setup_model_and_optimizer(args,**{})
# model=model.module
args.load=args.load_pretrained
load_checkpoint(model, optimizer, lr_scheduler, args)


# imp.reload(zj_finetune_glm)
# text='The American House of Representatives is due to vote today, on a resolution submitted by the Chairman of the Judicial Committee, Henry Hyde, asking Congress to launch an investigation over charges brought forward by Independent Prosecutor-General Kenneth Star against President Clinton in the Monica Lewinsky affair.'
# test_lines=[text*10]
# zj_finetune_glm.test(args,{},test_lines,model=model)


import os
from methods import pvp
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from methods.wrapper import MODEL_CLASSES
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
task_name='rte'
data_dir='/data/zhoujing/NLP/data/FewGLUE'
train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]),TRAIN_SET,num_examples=-1)
imp.reload(zj_finetune_glm)
gen_func=zj_finetune_glm.get_generate_data(task_name,train_examples=train_examples,priming_model='GLM',version=2)
source_texts,tgt_texts=gen_func(aug_num=1,mask_ratio=0.3,priming=True,priming_with_itself=True)


total_predictions,pred_blanks=zj_finetune_glm.test(args,{},(source_texts[0]['No'],tgt_texts[0]['No']),model=model)
'''
