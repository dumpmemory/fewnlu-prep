'''
T5 finetune
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb

model.generate: code in https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L101
'''

import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
import re
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd
import numpy as np
import random
import argparse
import copy
from genaug import gen_aug_T5
import imp
imp.reload(gen_aug_T5)
import string
# gen with only priming, and without finetune
# def gen_with_priming(converted_data,pattern_strings,model,tokenizer):
#	 strings_to_be_generated=
#	 t5_generate(strings_to_be_generated)
'''
from genaug import gen_aug_pattern
imp.reload(gen_aug_pattern)
data=gen_aug_pattern.get_finetune_priming_data(converted_data,pattern_strings,equal_mix=True)

imp.reload(gen_aug_pattern)
args=gen_aug_pattern.init_t5_args()
imp.reload(gen_aug_pattern)
new_model=gen_aug_pattern.t5_finetune(args,data,model,tokenizer)

'''
import os
from methods import pvp
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from methods.wrapper import MODEL_CLASSES
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
import torch
# data_dir='/mnt3/FewGLUE'
# task_name='rte'
# train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]) , TRAIN_SET,num_examples=-1)

total_pattern_nums={'rte':4,'copa':2,'wsc':3,'cb':4,'wic':3,'boolq':6,'multirc':3,'record':1}
class TmpWrapper():
    def __init__(self,config,tokenizer):
        self.config=config
        self.tokenizer=tokenizer

def init_virtual_tokenizer_wrapper():
	config_class = MODEL_CLASSES['albert']['config']
	tokenizer_class = MODEL_CLASSES['albert']['tokenizer']
	config=config_class.from_pretrained('albert-xxlarge-v2')
	config.wrapper_type='cls'
	tokenizer=tokenizer_class.from_pretrained('albert-xxlarge-v2')
	wrapper=TmpWrapper(config,tokenizer)
	return tokenizer,wrapper

def get_primming_data(task_name,data_dir='/data/zhoujing/NLP/data/FewGLUE',train_examples=None,priming_model='T5'):
	if priming_model=='T5':
		substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(10)]
	else:
		substitute_verbalizers=['<blank>']*100
	assert data_dir is not None or train_examples is not None
	if train_examples is None:
		train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]),TRAIN_SET,num_examples=-1)
	tokenizer,wrapper=init_virtual_tokenizer_wrapper()
	converted_data={};pattern_strings={};according_examples={}
	for pattern_id in range(total_pattern_nums[task_name]):
		mypvp=pvp.PVPS[task_name](wrapper,pattern_id,None)
		converted_data[pattern_id]={};according_examples[pattern_id]={}
		pattern_strings[pattern_id]={}
		# import pdb
		# pdb.set_trace()
		for (i,example) in enumerate(train_examples):
			# print(i)
			parts_a, parts_b=mypvp.get_parts(example)
			parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
			parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
			label=mypvp.verbalize(example.label)[0]
			if label not in converted_data[pattern_id]:
				converted_data[pattern_id][label]=[]
				according_examples[pattern_id][label]=[]
				tmp=[];substitute_idx=0
				for x in parts_a+parts_b:
					if x[1]==True:
						tmp.append(substitute_verbalizers[substitute_idx])
						substitute_idx+=1
					else:
						if x[0].startswith('[MASK]'):
							tmp.append(label)
						else:
							tmp.append(x[0])
				pattern_strings[pattern_id][label]=tmp
			# converted_data[pattern_id][label].append("".join([label if x[0].startswith('[MASK]') else x[0] for x in parts_a+parts_b]))
			converted_data[pattern_id][label].append([label if x[0].startswith('[MASK]') else x[0] for x in parts_a+parts_b])
			according_examples[pattern_id][label].append(example)
			# if i==0:
			#	 candidate_labels=list(set([mypvp.verbalize(e.label)[0] for e in train_examples]))
			#	 for label in candidate_labels:
			#		 tmp=[];substitute_idx=0
			#		 for x in parts_a+parts_b:
			#			 if x[1]==True:
			#				 tmp.append(substitute_verbalizers[substitute_idx])
			#				 substitute_idx+=1
			#			 else:
			#				 if x[0].startswith('[MASK]'):
			#					 tmp.append(label)
			#				 else:
			#					 tmp.append(x[0])
			#		 pattern_strings[pattern_id].append(tmp)
	return converted_data,pattern_strings,according_examples
	
def get_finetune_priming_data(converted_data,pattern_strings,pattern_id=0,priming_num=2,pattern_position=0,examples_num=100,equal_mix=True,sep_token='</s>'):
	# priming_num=4;pattern_position=0;examples_num=100
	data=[]
	if equal_mix==True:
		# import pdb
		# pdb.set_trace()
		priming_num=len(converted_data[pattern_id].keys())
		for _ in range(examples_num):
			is_extra=False;extra_id=0
			train_text,label,total_text=[],[],[]
			if pattern_position==-1:
				pattern_pos=np.random.choice(priming_num)
			else:
				pattern_pos=min(pattern_position,priming_num)
			key_list=list(converted_data[pattern_id].keys())
			random.shuffle(key_list)
			for (i,la) in enumerate(key_list):
				text=converted_data[pattern_id][la][np.random.choice(range(len(converted_data[pattern_id][la])))]
				total_text+=text
				if i==pattern_pos:
					train_text+=pattern_strings[pattern_id][la]
					for (word_idx,word) in enumerate(pattern_strings[pattern_id][la]):
						if word.startswith('<extra_id_'):
							if is_extra==True:
								label.append('<extra_id_{}>'.format(extra_id))
								extra_id+=1
							label.append(text[word_idx])
							is_extra=False
						else:
							is_extra=True
				else:
					# train_text.append(text)
					train_text+=text
					is_extra=True
				
				if i!=len(list(converted_data[pattern_id].keys()))-1:
					train_text.append(sep_token)
					total_text.append(sep_token)
					# label.append('</s>')
				# print(train_text)
			if is_extra==True:
				label.append('<extra_id_{}>'.format(extra_id))
			total_text.append('</s>')
			train_text.append('</s>')
			label.append('</s>')
			data.append({'train_text':' '.join(train_text),'label':' '.join(label),'total_text':' '.join(total_text)})
	else:
		for _ in range(examples_num):
			is_extra=False;extra_id=0
			train_text,label,total_text=[],[],[]
			idxs=[]
			# import pdb
			# pdb.set_trace()
			for _ in range(priming_num):
				idx1=np.random.choice(list(converted_data[pattern_id].keys()))
				idx2=np.random.choice(range(len(converted_data[pattern_id][idx1])))
				idxs.append((idx1,idx2))
			# idxs=np.random.choice(range(len(converted_data[pattern_id])),priming_num,False)
			if pattern_position==-1:
				pattern_pos=np.random.choice(range(priming_num))
			else:
				pattern_pos=min(pattern_position,priming_num-1)
			for i in range(priming_num):
				text=converted_data[pattern_id][idxs[i][0]][idxs[i][1]]
				total_text+=text
				if i==pattern_pos:
					# train_text+=pattern_strings[pattern_id][np.random.choice(range(len(pattern_strings[pattern_id])))]
					la=idxs[i][0]
					train_text+=pattern_strings[pattern_id][la]
					for (word_idx,word) in enumerate(pattern_strings[pattern_id][la]):
						if word.startswith('<extra_id_'):
							if is_extra==True:
								label.append('<extra_id_{}>'.format(extra_id))
								extra_id+=1
							label.append(text[word_idx])
							is_extra=False
						else:
							is_extra=True
				else:
					train_text+=text
					is_extra=True
				if i!=priming_num-1:
					train_text.append(sep_token)
					total_text.append(sep_token)
			if is_extra==True:
				label.append('<extra_id_{}>'.format(extra_id))
			total_text.append('</s>')
			train_text.append('</s>')
			label.append('</s>')
			data.append({'train_text':' '.join(train_text),'label':' '.join(label),'total_text':' '.join(total_text)})
	return data


def get_generate_data(task_name,data_dir='/data/zhoujing/NLP/data/FewGLUE',train_examples=None,priming_model='T5',version=2,split='train'):
	#############################################################################################################################
	# version:
	#   1: priming version, mask text_b
	#   2: only mask the first sentence, with mask_ratio, (then fill it to length 512)
	#   3: 
	# if split=='train': return (source_texts,target_texts)
	# else: return source_texts
	#############################################################################################################################
	if priming_model=='T5':
		substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)]
	else:
		substitute_verbalizers=['<blank>']*300
	assert data_dir is not None or train_examples is not None
	if train_examples is None:
		train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]),TRAIN_SET,num_examples=-1)
	tokenizer,wrapper=init_virtual_tokenizer_wrapper()
	total_pattern_nums={'rte':4,'copa':1,'wsc':3,'cb':4,'wic':3,'boolq':6,'multirc':3,'record':1}
	def generate_v1():
		converted_data={};pattern_strings={};according_examples={}
		for pattern_id in range(total_pattern_nums[task_name]):
			mypvp=pvp.PVPS[task_name](wrapper,pattern_id,None)
			converted_data[pattern_id]={};according_examples[pattern_id]={}
			pattern_strings[pattern_id]={}
			# import pdb
			# pdb.set_trace()
			for (i,example) in enumerate(train_examples):
				# print(i)
				parts_a, parts_b=mypvp.get_parts(example)
				parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
				parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
				label=mypvp.verbalize(example.label)[0]
				if label not in converted_data[pattern_id]:
					converted_data[pattern_id][label]=[]
					according_examples[pattern_id][label]=[]
					tmp=[];substitute_idx=0
					for x in parts_a+parts_b:
						if x[1]==True:
							tmp.append(substitute_verbalizers[substitute_idx])
							substitute_idx+=1
						else:
							if x[0].startswith('[MASK]'):
								tmp.append(label)
							else:
								tmp.append(x[0])
					pattern_strings[pattern_id][label]=tmp
				# converted_data[pattern_id][label].append("".join([label if x[0].startswith('[MASK]') else x[0] for x in parts_a+parts_b]))
				converted_data[pattern_id][label].append([label if x[0].startswith('[MASK]') else x[0] for x in parts_a+parts_b])
				according_examples[pattern_id][label].append(example)
		return converted_data,pattern_strings,according_examples

	def generate_v2(aug_num=10,mask_ratio=0.5,priming=False, priming_length=512, priming_with_itself=True): 
		# we do not mask pattern, we just mask text_a and text_b
		# if priming is True, we will append priming examples until length priming_length
		source_texts={};target_texts={};according_examples={};pure_parts={}
		organized_examples={}
		for pattern_id in range(total_pattern_nums[task_name]):
			source_texts[pattern_id]={};target_texts[pattern_id]={};according_examples[pattern_id]={};pure_parts[pattern_id]={}
			if task_name=='copa':
				source_texts[pattern_id]={}
				organized_examples[pattern_id]={}
				for (i,example) in enumerate(train_examples[:32]):
					question_label=example.meta['question']
					label=int(example.label)
					parts=[]
					possible_answers=[example.meta['choice1'].rstrip(string.punctuation),example.meta['choice2'].rstrip(string.punctuation)]
					parts.append((example.text_a.rstrip(string.punctuation), True))
					if example.meta['question']=='cause':
						parts.append((', because', False))
					else:
						parts.append((', so',False))
					parts.append((possible_answers[label],True))
					parts.append((', but not',False))
					parts.append((possible_answers[1-label],True))

					if question_label not in organized_examples[pattern_id]:
						source_texts[pattern_id][question_label]=[]
						target_texts[pattern_id][question_label]=[]
						organized_examples[pattern_id][question_label]=[]
						according_examples[pattern_id][question_label]=[]
						pure_parts[pattern_id][question_label]=[]
					organized_examples[pattern_id][question_label].append(parts)
			else:
				mypvp=pvp.PVPS[task_name](wrapper,pattern_id,None)
				organized_examples[pattern_id]={}
				for (i,example) in enumerate(train_examples):
					if task_name=='multirc':
						parts_a,parts_b=mypvp.get_parts(example,total_shortenable=True)
					else:
						parts_a,parts_b=mypvp.get_parts(example)
					parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
					parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
					label=mypvp.verbalize(example.label)[0]
					if label not in organized_examples[pattern_id]:
						source_texts[pattern_id][label]=[]
						target_texts[pattern_id][label]=[]
						organized_examples[pattern_id][label]=[]
						according_examples[pattern_id][label]=[]
						pure_parts[pattern_id][label]=[]
					organized_examples[pattern_id][label].append(parts_a+parts_b)

		def mask_text(text,cnt=0): #jing, add pred
			tokens = text.split()
			n = len(tokens)
			indices = sorted(random.sample(range(n), int(n * mask_ratio)))
			masked_src, masked_tgt = "", []
			for i, idx in enumerate(indices):
				if i == 0 or idx != indices[i-1] + 1:
					masked_tgt.append("")
				masked_tgt[-1] += " " + tokens[idx]
				tokens[idx] = "[MASK]"
			for i, token in enumerate(tokens):
				if i != 0 and token == "[MASK]" and tokens[i-1] == "[MASK]":
					continue
				if token=="[MASK]":
					masked_src+=" "+substitute_verbalizers[cnt]
					cnt+=1
				else:
					masked_src += " " + token
			# print(masked_src,masked_tgt)
			return masked_src, masked_tgt, cnt

		for (pattern_id,organized_example) in organized_examples.items():
			for (i,(label,parts_list)) in enumerate(organized_example.items()):
				# print(parts_list)
				for parts in parts_list:
					pure_parts[pattern_id][label].append([])
					for aug_id in range(aug_num):
						source_text='';target_text=[]
						cnt=0
						for part in parts:
							# print(part)
							if part[1]==False:
								if part[0].startswith('[MASK]'):
									source_text+=label
									# print('sess')
								else:
									source_text+=part[0]
							else:
								masked_src,masked_tgt,cnt=mask_text(part[0],cnt=cnt)
								pure_parts[pattern_id][label][-1].append(masked_src)
								source_text+=masked_src
								target_text+=masked_tgt
						if priming==True:
							# import pdb
							# pdb.set_trace()
							total_length=len(source_text.split())
							if priming_with_itself==True:
								while(total_length<priming_length):
									add_src=' '.join([label if x[0].startswith('[MASK]') else x[0] for x in parts])
									source_text+=add_src
									total_length+=len(add_src.split())
							else:
								augidx=list(range(len(parts_list)))
								random.shuffle(augidx)
								for idx in augidx:
									if total_length>=priming_length: break
									if idx==i: continue
									add_src=' '.join([label if x[0].startswith('[MASK]') else x[0] for x in organized_example[label][idx]])
									source_text+=add_src
									total_length+=len(add_src.split())
						source_texts[pattern_id][label].append(source_text)
						target_texts[pattern_id][label].append(target_text)
						according_examples[pattern_id][label].append(' '.join([label if part[0].startswith('[MASK]') else part[0] for part in parts]))
		return source_texts,target_texts,according_examples,pure_parts
	if version==1:
		return generate_v1
	elif version==2:
		return generate_v2


# imp.reload(gen_aug_pattern)
# gen_aug_pattern.generate_pattern_with_priming(converted_data,pattern_strings,model,tokenizer)
def generate_pattern_with_priming(converted_data,pattern_strings,model,tokenizer,priming_num=2,pattern_position=0,equal_mix=True,pattern_id=0,beam=10,gen_type='default',max_gen_length=20):
	data=get_finetune_priming_data(converted_data,pattern_strings,pattern_id=pattern_id,priming_num=priming_num,pattern_position=pattern_position,examples_num=10,equal_mix=equal_mix)
	# print('input sentences:',strings_to_be_generated)
	# import pdb
	# pdb.set_trace()
	model.eval()
	for x in data:
		strings_to_be_generated=x['train_text']
		print('>> input_sentences:\n',strings_to_be_generated)
		if gen_type=='default':
			input=tokenizer(''.join(strings_to_be_generated),return_tensors='pt')
			if beam==1: num_return_sequences=1;do_sample=True
			else: num_return_sequences=int(np.ceil(beam//2));do_sample=False
			predected_results=model.generate(input_ids=input.input_ids.cuda(),attention_mask=input.attention_mask.cuda(),
			max_length=max_gen_length,num_beams=beam,num_return_sequences=num_return_sequences,do_sample=do_sample,
			repetition_penalty=2.5,length_penalty=1.0,early_stopping=True)
			for (i,ans) in enumerate(predected_results):
				print('>> my_prediction_{}:\n'.format(i),tokenizer.decode(ans))
			print('')
		else:
			generate_text=t5_generate([strings_to_be_generated],model, tokenizer, target_number=2, beam=2, label=None,max_gen_length=max_gen_length,truncate='head')[:beam//2]


# def augment_sentences(strings_to_be_generated,model,tokenizer,beam=10,gen_type='default',max_gen_length=100,num_return_sequences=None,verbose=False,model_type='t5',aug_type='rand_iter'):
# 	if model_type!='t5':
# 		print('not implemented')
# 		return
# 	if aug_type=='direct':
# 		total_predictions, pred_blanks=gen_aug_T5.generate_t5_blanks(strings_to_be_generated,model,tokenizer,num_return_sequences=1)
# 	elif aug_type=='rand_iter':


def recover_examples_from_blanks(pure_parts,pred_blanks,model_type=None):
	# example_lines=[['[MASK] x','[MASK] y'],['x [MASK] y', '[MASK] z']]
	# pred_blanks=[['a','b'],['c','d']]
	# return filled_parts=[['a x','b y'],['x c y','d z']]
	if model_type is None:
		lines=' '.join([' '.join(x) for x in pure_parts])
		if '[MASK]' in lines:
			model_type='GLM'
		elif '<extra_id_0>' in lines:
			model_type='t5'
	# import pdb
	# pdb.set_trace()
	filled_parts=[]
	for (parts,pred_blank) in zip(pure_parts,pred_blanks):
		current_blank=0
		filled_parts.append([])
		for part in parts:
			output_tokens=[]
			tokens=part.split()
			for token in tokens:
				if (model_type.lower()=='t5' and token.startswith('<extra_id_')) or (model_type.lower=='glm' and token.startswith('[MASK]')):
					if current_blank < len(pred_blank):
						output_tokens.append(pred_blank[current_blank])
					current_blank+=1
				else:
					output_tokens.append(token)
			filled_parts[-1].append(' '.join(output_tokens))
	return filled_parts


def save_recovered_examples(example_pattern,filled_parts,key_names=('text_b','text_a','label'),save_path=''):
	augmented_examples=[]
	for filled_part in filled_parts:
		new_example=copy.deepcopy(example_pattern)
		for (x,key_name) in zip(filled_part,key_names):
			if ':' in key_name:
				[key_name1,key_name2]=key_name.split(":")
				new_example.__dict__[key_name1][key_name2]=x
			else:
				new_example.__dict__[key_name]=x
		augmented_examples.append(new_example)
	torch.save(augmented_examples,save_path)
	return augmented_examples

############jing warning warning
def aug_wsc_lonp(example,gen_blanks_func,model,tokenizer,aug_num=10,mask_ratio=0.5):
	# mask 50% of the sentence (except the nouns and pronouns) and then fill in
	substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)]
	#################################### mask the text ##########################################
	################### processed results: masked_src, masked_tgt, cnt, new_span1_index, new_span2_index ##########################
	tokens=example.text_a.split()
	span1_index=example.meta['span1_index']
	span2_index=example.meta['span2_index']
	span1_text=example.meta['span1_text']
	span2_text=example.meta['span2_text']
	slen1=len(span1_text.split())
	slen2=len(span2_text.split())
	n=len(tokens)
	cnt=0
	indices=sorted(random.sample(range(n),int(n*mask_ratio)))
	splited_masked_src=[]
	for i,idx in enumerate(indices):
		if idx in range(span1_index,span1_index+slen1) or idx in range(span2_index,span2_index+slen2):
			continue
		tokens[idx]="[MASK]"

	for i, token in enumerate(tokens):
		if i==span1_index:
			new_span1_index=len(splited_masked_src)
		if i==span2_index:
			new_span2_index=len(splited_masked_src)
		if i != 0 and token == "[MASK]" and tokens[i-1] == "[MASK]":
			continue
		if token=="[MASK]":
			splited_masked_src.append(substitute_verbalizers[cnt])
			cnt+=1
		else:
			splited_masked_src.append(token)
	masked_src=' '.join(splited_masked_src)
	################################## predict with T5 ##################################
	total_predictions,preds=gen_blanks_func([masked_src],model,tokenizer,max_gen_length=40,num_return_sequences=1)
	pred_blank=preds[0]
	model_type='t5'
	current_blank=0
	output_tokens=[]
	new_cnt=0
	for (i,token) in enumerate(splited_masked_src):
		if i==new_span1_index:
			span1_index=new_cnt
		if i==new_span2_index:
			span2_index=new_cnt
		if (model_type.lower()=='t5' and token.startswith('<extra_id_')) or (model_type.lower()=='glm' and token.startswith('[MASK]')):
			if current_blank <len(pred_blank):
				output_tokens.append(pred_blank[current_blank])
				new_cnt+=len(pred_blank[current_blank].split())
			current_blank+=1
		else:
			output_tokens.append(token)
			new_cnt+=1

	new_example=copy.deepcopy(example)
	new_example.__dict__['text_a']=' '.join(output_tokens)
	new_example.__dict__['meta']['span1_index']=span1_index
	new_example.__dict__['meta']['span2_index']=span2_index
	return new_example




def aug_wsc(task_name,train_examples,gen_blanks_func,model,tokenizer,aug_num=10,mask_ratio=0.5,save_path=None, priming_model='T5',aug_type='lonp'):
	if priming_model=='T5':
		substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)]
	else:
		substitute_verbalizers=['<blank>']*300
	if aug_type=='lonp':
		augmented_examples=[]
		for _ in range(aug_num):
			for e in train_examples:
				new_example=aug_wsc_lonp(e,gen_blanks_func,model,tokenizer,aug_num=10,mask_ratio=0.5)
				augmented_examples.append(new_example)
	elif aug_type=='onlynp':
		augmented_examples=[]
		for e in train_examples:
			splited_text=e.text_a.split()
			span1_index=e.meta['span1_index']
			span2_index=e.meta['span2_index']
			span1_text=e.meta['span1_text']
			span2_text=e.meta['span2_text']
			slen1=len(span1_text.split())
			slen2=len(span2_text.split())
			new_example=copy.deepcopy(e)
			text_to_be_augmented=[]
			if span1_index<span2_index:
				text_to_be_augmented+=(splited_text[:span1_index])
				text_to_be_augmented.append(substitute_verbalizers[0])
				text_to_be_augmented+=(splited_text[span1_index+slen1:span2_index])
				text_to_be_augmented.append(substitute_verbalizers[1])
				text_to_be_augmented+=(splited_text[span2_index+slen2:])
				print(text_to_be_augmented)
				text_to_be_augmented=' '.join(text_to_be_augmented)
				total_predictions,preds=gen_blanks_func([text_to_be_augmented],model,tokenizer,max_gen_length=20,gen_type='miao',num_return_sequences=1)
				new_example.__dict__['text_a']=total_predictions[0]
				new_example.__dict__['meta']['span1_text']=preds[0][0]
				new_example.__dict__['meta']['span2_text']=preds[0][1]
				new_example.__dict__['meta']['span1_index']=span1_index
				new_example.__dict__['meta']['span2_index']=span2_index+len(preds[0][0].split())-slen1
				augmented_examples.append(new_example)
			else:
				text_to_be_augmented+=(splited_text[:span2_index])
				text_to_be_augmented.append(substitute_verbalizers[0])
				text_to_be_augmented+=(splited_text[span2_index+slen2:span1_index])
				text_to_be_augmented.append(substitute_verbalizers[1])
				text_to_be_augmented+=(splited_text[span1_index+slen1:])
				print(text_to_be_augmented)
				text_to_be_augmented=' '.join(text_to_be_augmented)
				total_predictions,preds=gen_blanks_func([text_to_be_augmented],model,tokenizer,max_gen_length=20,gen_type='miao',num_return_sequences=1)
				new_example.__dict__['text_a']=total_predictions[0]
				new_example.__dict__['meta']['span1_text']=preds[0][1]
				new_example.__dict__['meta']['span2_text']=preds[0][0]
				new_example.__dict__['meta']['span1_index']=span2_index
				new_example.__dict__['meta']['span2_index']=span1_index+len(preds[0][1].split())-slen2
				augmented_examples.append(new_example)
	torch.save(augmented_examples,os.path.join(save_path,'t5_0.5_{}_wsc'.format(aug_type)))
	return augmented_examples



def aug_general(task_name,train_examples,gen_blanks_func,model,tokenizer,pattern_id=0,aug_num=10,mask_ratio=0.5,save_path=None,key_names=('text_b','text_a','label'),aug_type='rand_iter_5'):
	if task_name=='wsc':
		return aug_wsc(task_name,train_examples,gen_blanks_func,model,tokenizer,aug_num,mask_ratio,save_path,priming_model='T5')
	gen_func=get_generate_data(task_name,train_examples=train_examples,priming_model='T5',version=2)
	if task_name=='rte':
		key_map={'No':'not_entailment','Yes':'entailment'}
	elif task_name=='boolq':
		pattern_id=4
		key_map={'No':'False','Yes':'True'}
	elif task_name=='cb':
		pattern_id=2
		key_map={'No':'contradiction','Yes':'entailment','Maybe':'neutral'}
	elif task_name=='copa':
		key_map={'cause':'cause','effect':'effect'}
		key_names=('text_a','meta:choice1','meta:choice2','meta:question')
	elif task_name=='multirc':
		pattern_id=0
		key_map={'No':'0','Yes':'1'}
		key_names=('text_a','text_b','meta:answer','label')
	total_filled_parts=[]
	for _ in range(aug_num):
		source_texts, tgt_texts, according_examples, pure_parts = gen_func(aug_num=1, mask_ratio=mask_ratio, priming=False)
		# print(source_texts,pure_parts)
		for key in source_texts[pattern_id].keys():
			if aug_type.startswith('rand_iter'):
				batch_size=int(aug_type.split('_')[2])
				texts_to_be_augmented=source_texts[pattern_id][key]
				pred_blanks=[]
				for (text_to_be_augmented,tgt_parts) in zip(texts_to_be_augmented,tgt_texts[pattern_id][key]):
					blen=len(tgt_parts)
					new_tgt_parts=copy.deepcopy(tgt_parts)
					masked_idxs=list(range(blen))
					random.shuffle(masked_idxs)
					text_parts=re.split('<extra_id_\d+>',text_to_be_augmented)
					for batch_idx in range(int(np.ceil(len(masked_idxs)/batch_size))):
						cnt=0
						masked_id=masked_idxs[batch_idx*batch_size:(batch_idx+1)*batch_size]
						masked_id=sorted(masked_id)
						new_text=''
						for i in range(len(text_parts)):
							new_text+=text_parts[i]
							if i!=blen:
								if i in masked_id:
									new_text+='<extra_id_{}>'.format(cnt)
									cnt+=1
								else:
									new_text+=new_tgt_parts[i]
						# print(new_text)
						total_predictions,preds=gen_blanks_func([new_text],model,tokenizer,num_return_sequences=1)
						preds=preds[0]
						if len(preds)>len(masked_id):
							preds=preds[:len(masked_id)]
						else:
							for _ in range(len(masked_id)-len(preds)):
								preds.append('')
						# print(total_predictions,masked_id,preds)
						for (m_id,pred_blank) in zip(masked_id,preds):
							# print(m_id,pred_blank)
							new_tgt_parts[m_id]=pred_blank
					pred_blanks.append(new_tgt_parts)
					# total_predictions, pred_blanks=gen_blanks_func(text_to_be_augmented,model,tokenizer,num_return_sequences=1)
			elif aug_type=='default':
				total_predictions, pred_blanks=gen_blanks_func(source_texts[pattern_id][key],model,tokenizer,num_return_sequences=1)
			filled_parts=recover_examples_from_blanks(pure_parts[pattern_id][key],pred_blanks)
			filled_parts=[x+[key_map[key]] for x in filled_parts]
			if task_name=='copa':
				filled_parts=[x+['0'] for x in filled_parts]
			# print(filled_parts)
			total_filled_parts+=filled_parts
	
	if aug_type=='default':
		save_path=os.path.join(save_path,'t5_{}_{}'.format(mask_ratio,task_name))
	elif aug_type.startswith('rand_iter'):
		save_path=os.path.join(save_path,'t5_{}_{}_{}'.format(mask_ratio,aug_type,task_name))
	augmented_examples=save_recovered_examples(train_examples[pattern_id],total_filled_parts,key_names=key_names,save_path=save_path)
	return augmented_examples



'''
import os
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
data_dir='/data/zhoujing/NLP/data/FewGLUE'
task_name='rte'
train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]) , TRAIN_SET,num_examples=-1)
unlabeled_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), UNLABELED_SET, num_examples=-1)
test_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), DEV_SET, num_examples=-1)

from genaug import gen_aug_T5, gen_aug_pattern
t5_args=gen_aug_T5.init_t5_args()
tokenizer,model=gen_aug_T5.init_model()

augmented_examples=gen_aug_pattern.aug_rte(train_examples,gen_aug_T5.generate_t5_blanks,model,tokenizer,aug_num=10,mask_ratio=0.5,save_path='results/augmented_examples/rte_0.5')



'''