'''
We want to use finetuned model to filter bad augmented examples 

'''
from tqdm import trange,tqdm
import torch
import transformers

from transformers import AlbertForMaskedLM
from methods.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import itertools
import numpy as np
import copy

'''
path='/data/zhoujing/NLP/augframe/results/augmented_examples/t5_0.5_rte'
import torch
examples=torch.load(path)
from utils import confidence_filter
import imp
imp.reload(confidence_filter)
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/data/zhoujing/NLP/baseline/PET/PET_2.0.0_myself/results/several_seed_ensemble/methods/rte_32_albert_model/seed_0/p1-i0')
import methods
eval_config = methods.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=8, decoding_strategy='default', priming=False)
# myfilter.validate(myfilter.wrapper,1,examples,eval_config)
new_examples=[]
for batch_idx in range(10):
	new_examples.append(examples[batch_idx*32:(batch_idx+1)*32])


filtered_examples=myfilter.recover_labels(myfilter.wrapper,1,new_examples,eval_config,recover_type='max_eachla')
'''

class Confidence_Filter(object):
	def __init__(self,pattern_iter_output_dir=None,wrapper=None):
		assert pattern_iter_output_dir is None or wrapper is None
		self.wrappers=None
		self.wrapper=None
		if pattern_iter_output_dir is not None:
			self.wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
		if wrapper is not None:
			self.wrapper=wrapper

	def reload_wrapper(self,wrapper=None,pattern_iter_output_dir=None):
		if wrapper is not None:
			self.wrapper=wrapper
		else:
			if isinstance(pattern_iter_output_dir,list):
				self.wrappers=[]
				for path in pattern_iter_output_dir:
					self.wrappers.append(TransformerModelWrapper.from_pretrained(path))
			else:
				self.wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

	def validate(self,wrapper,pattern_id,eval_data,eval_config):
		wrapper.model.to(eval_config.device)
		if isinstance(eval_data,list):
			total_data=[]
			for d in eval_data:
				total_data+=d
			output=wrapper.eval(total_data,eval_config.device,per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
				n_gpu=eval_config.n_gpu,decoding_strategy=eval_config.decoding_strategy,priming=eval_config.priming)
		else:
			output=wrapper.eval(eval_data,eval_config.device,per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
				n_gpu=eval_config.n_gpu,decoding_strategy=eval_config.decoding_strategy,priming=eval_config.priming)
		return output

	def recover_labels(self,wrapper,pattern_id,eval_data,eval_config,recover_type='max_prevla'):
		# eval_data:[[eval_data1],[eval_data2],...,[eval_datan]]
		# recover_type: 
			# 'max_prevla': select the most likely label which preserve its label
			# 'max_eachla': for each label, select the most likely one
		aug_num=len(eval_data)
		example_num=len(eval_data[0])
		output=self.validate(wrapper,pattern_id,eval_data,eval_config)
		label_num=output['logits'].shape[1]
		return_examples=[]
		label_map=self.wrapper.preprocessor.label_map
		inverse_label_map={x:y for (y,x) in label_map.items()}
		if recover_type=='max_prevla':
			for i in range(example_num):
				logits=output['logits'][np.array([_*example_num+i for _ in range(aug_num)])]
				prevla=output['labels'][i].item()
				max_idx=np.argmax(logits[:,prevla])
				return_examples.append(eval_data[max_idx][i])
		elif recover_type=='max_eachla':
			eval_data=list(itertools.chain.from_iterable(eval_data))
			# import pdb
			# pdb.set_trace()
			for i in range(example_num):
				logits=output['logits']
				max_pos=np.ones(label_num,dtype=int)*(-1)
				for _ in range(aug_num):
					logit=logits[_*example_num+i]
					la_now=np.argmax(logit)
					if max_pos[la_now]==-1 or logit[la_now]>logits[max_pos[la_now]][la_now]:
						max_pos[la_now]=_*example_num+i
				for (la,pos) in enumerate(max_pos):
					if pos==-1:
						continue
					new_example=copy.deepcopy(eval_data[pos])
					new_example.label=inverse_label_map[la]
					return_examples.append(new_example)
		return return_examples



	def choose_most_possible_samples(self,pattern_id,eval_data,eval_config):
		# eval_data:[[eval_data1],[eval_data2],...,[eval_datan]]
		output=self.validate(pattern_id,list(itertools.chain.from_iterable(eval_data)),eval_config)
		logits=output['logits']
		aug_len=len(eval_data)
		dlen=len(eval_data[0])
		ans=[]
		for i in range(dlen):
			max_pos=-1;max_score=-100
			for j in range(aug_len):
				score=logits[dlen*j+i]
				if j==0 or score>max_score:
					max_score=score
					max_pos=j
			ans.append(eval_data[j][i])
		return ans

	def del_finetuned_model(self):
		if self.wrappers is not None:
			for i in range(len(self.wrappers)):
				self.wrappers[i].model=None
				self.wrappers[i]=None
			torch.cuda.empty_cache()
		else:
			self.wrapper.model = None
			self.wrapper = None
			torch.cuda.empty_cache()


from methods.wrapper import TRAIN_STEP_FUNCTIONS,TransformerModelWrapper
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
class Gradient_Marker(object):
	def __init__(self,pattern_iter_output_dir=None):
		if pattern_iter_output_dir is not None:
			self.wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

	def reload_wrapper(self,wrapper):
		self.wrapper=wrapper

	def get_word_gradient(self,data,device='cuda:0'):
		self.wrapper.model.to(device)
		dataset = self.wrapper._generate_dataset(data, priming=False)
		batch_size = 1
		sampler = SequentialSampler(dataset)
		dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
		epoch_iterator=tqdm(dataloader,desc='Iteration')
		for step,batch in enumerate(epoch_iterator):
			self.wrapper.model.train()
			unlabeled_batch=None
			batch={k:t.to(device) for k,t in batch.items()}
			train_step_inputs = {
				'unlabeled_batch': None, 'lm_training': False, 'alpha': 0,
				'use_logits': False, 'temperature': 1
			}
			loss = self.wrapper.task_helper.train_step(batch, **train_step_inputs) if self.wrapper.task_helper else None
			if loss is None:
				loss = TRAIN_STEP_FUNCTIONS[self.wrapper.config.wrapper_type](self.wrapper)(batch, **train_step_inputs)
			loss.backward()
			import pdb
			pdb.set_trace()
			# calculate the norm of each word, and find the maximum
			grad_norm=torch.norm(self.wrapper.model.albert.embeddings.word_embeddings.weight.grad,dim=0)
			grad_norm_mean=grad_norm.mean().item()
			print(self.wrapper.tokenizer.decode(batch['input_ids'][0]))
			for i,norm in enumerate(grad_norm):
				if norm.item()>grad_norm_mean: print(self.wrapper.tokenizer._convert_id_to_token(batch['input_ids'][0][i].item()))
			self.wrapper.model.zero_grad()
			torch.cuda.empty_cache()

	def del_finetuned_model(self):
		self.model=None
		torch.cuda.empty_cache()

'''
pattern_iter_output_dir='results/pet_final/rte_32_albert_model'
wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)




from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
task_name='boolq'
processor = PROCESSORS[task_name]()
data_dir="/home/zhoujing/NLP/data/FewGLUE/{}".format(NAME[task_name])
train_examples = load_examples(task_name, data_dir, TRAIN_SET, num_examples=-1)
unlabeled_examples = load_examples(task_name, data_dir, UNLABELED_SET, num_examples=-1)
test_examples = load_examples(task_name, data_dir, DEV_SET, num_examples=-1)



# from utils import confidence_filter
# import imp
# imp.reload(confidence_filter)
# pattern_iter_output_dir='/home/zhoujing/NLP/baseline/PET/PET_2.0.0_myself/results/pet_final/boolq_32_albert_model/p0-i0'
# model=confidence_filter.Gradient_Marker.from_pretrained(pattern_iter_output_dir)
# model.get_word_gradient(train_examples)


from utils import confidence_filter
import imp
imp.reload(confidence_filter)
pattern_iter_output_dir='/home/zhoujing/NLP/baseline/PET/PET_2.0.0_myself/results/pet_final/boolq_32_albert_model/p0-i0'
from methods.wrapper import TRAIN_STEP_FUNCTIONS,TransformerModelWrapper
wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)


imp.reload(confidence_filter)
model=confidence_filter.Gradient_Marker()
model.reload_wrapper(wrapper)
model.get_word_gradient(unlabeled_examples,device='cpu')
'''