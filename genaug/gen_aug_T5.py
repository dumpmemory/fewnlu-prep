import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd
import numpy as np
import random
import argparse


def init_t5_args(parser=None):
	if parser is None:
		parser=argparse.ArgumentParser()
	parser.add_argument("--t5_batch_size",default=2,type=int)
	parser.add_argument("--t5_learning_rate",default=1e-4,type=float)
	parser.add_argument("--t5_epochs",default=3,type=int)
	parser.add_argument("--device",default='cuda:0',type=str)
	parser.add_argument("--t5_gradient_accumulate_steps",default=1,type=int)
	return parser


def init_model():
	tokenizer = T5Tokenizer.from_pretrained('t5-large')
	tokenizer.sep_token = '</s>'
	model = T5ForConditionalGeneration.from_pretrained('t5-large')
	model=model.cuda()
	model.eval()
	return tokenizer,model
# t5_learning_rate
# t5_epochs
# device
# t5_gradient_accumulate_steps
# args=gen_aug_pattern.init_t5_args()
# new_model=gen_aug_pattern.t5_finetune(args,data,model,tokenizer)
def save_augmented_lines(augmented_lines):
	pass

def t5_finetune(args,data,model,tokenizer):
	# input_tensors=[]
	batch_size=args.t5_batch_size
	# input_texts=[x['train_text'] for x in data]
	# output_texts=[x.label for x in data]
	# input_ids=tokenizer(input_texts).input_ids
	# output_ids=tokenizer(output_texts).input_ids
	# input_ids = torch.zeros((len(input_ids), max_length)).long()
	# attention_mask = torch.zeros((len(input_ids), max_length)).long()
	# for i in range(len(input_tensors)):
	#	 input_ids[i, :input_tensors[i].size(-1)] = torch.tensor(input_tensors[i]).long()
	#	 attention_mask[i, :input_tensors[i].size(-1)] = 1
	# input_ids=tokenizer(input_texts,return_tensors='pt').input_ids
	optimizer=torch.optim.Adam(params=model.parameters(),lr=args.t5_learning_rate)
	# scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=10)
	turn=int(np.ceil(len(data)/batch_size))
	for epoch in range(args.t5_epochs):
		idxs=list(range(len(data)))
		random.shuffle(idxs)
		for step in tqdm(range(turn)):
			model.train()
			start=step*batch_size
			end=min((step+1)*batch_size,len(data))
			input_texts=[data[idx]['train_text'] for idx in idxs[start:end]]
			labels=[data[idx]['label'] for idx in idxs[start:end]]
			input_tensors=tokenizer(input_texts).input_ids
			label_tensors=tokenizer(labels).input_ids
			input_max_length=max([len(x) for x in input_tensors])
			label_max_length=max([len(x) for x in label_tensors])
			input_ids=torch.zeros((len(input_tensors),input_max_length)).long()
			attention_mask=torch.zeros((len(input_ids),input_max_length)).long()
			label_ids=torch.zeros((len(label_tensors),label_max_length)).long()
			for i in range(len(input_texts)):
				input_ids[i,:len(input_tensors[i])]=torch.tensor(input_tensors[i]).long()
				attention_mask[i,:len(input_tensors[i])]=1
				label_ids[i,:len(label_tensors[i])]=torch.tensor(label_tensors[i]).long()
			# import pdb
			# pdb.set_trace()
			# import pdb
			# pdb.set_trace()
		# for (step,idx) in enumerate(idxs):
			# input_ids=tokenizer(data[idx]['train_text'],return_tensors='pt').input_ids
			# labels=tokenizer(data[idx]['label'],return_tensors='pt').input_ids
			label_ids[label_ids==tokenizer.pad_token_id]=-100
			input_ids=input_ids.to(args.device)
			label_ids=label_ids.to(args.device)
			outputs=model(input_ids=input_ids,labels=label_ids)
			loss=outputs.loss
			if args.t5_gradient_accumulate_steps>1:
				loss=loss/args.t5_gradient_accumulate_steps
			loss.backward()
			# tr_loss+=loss.item()
			if (step+1)%args.t5_gradient_accumulate_steps==0 or step==len(idxs)-1:
				# torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
				optimizer.step()
				# scheduler.step()
				model.zero_grad()
	return model

# def t5_generate(strings_to_be_generated, model, tokenizer, target_number, choice_type='beam',beam=10, label=None, length_limit=None, max_gen_length=20, truncate='tail', verbose=False):
# 	"""
# 	choice_type: ['beam','max','prob']
# 	Generate templates based on given inputs
# 	label: Only use instances with this label (deprecated)
# 	length_limit: At least generate content as long as length_limit (deprecated)
# 	"""
# 	special_token_mapping = {'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id}
# 	for i in range(10):
# 		special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id("<extra_id_%d>" % (i))
	
# 	def enc(text):
# 		if isinstance(text,list):
# 			new_tokens=[]
# 			for part in text:
# 				if part in special_token_mapping:
# 					if part == 'cls' and 'T5' in type(tokenizer).__name__:
# 						# T5 does not have cls token
# 						continue
# 					new_tokens.append(special_token_mapping[part])
# 				else:
# 					new_tokens+=tokenizer.encode(part, add_special_tokens=False)
# 		else:
# 			new_tokens=tokenizer.encode(text)
# 		return new_tokens
# 	# import pdb
# 	# pdb.set_trace()
# 	input_texts=[];input_tensors=[];max_length=0
# 	target_numbers=[]
# 	for s in strings_to_be_generated:
# 		input_text=enc(s)
# 		input_text=np.array(input_text)
# 		print(input_text)
# 		# import pdb
# 		# pdb.set_trace()
# 		target_numbers.append(int(np.sum((input_text>=32000) & (input_text<=32099))))
# 		if truncate is not None:
# 			if truncate=='head':
# 				input_text=input_text[-256:]
# 			elif truncate=='tail':
# 				input_text=input_text[:256]
# 			else:
# 				raise NotImplementedError
# 			input_ids=torch.tensor(input_text).long()
# 			max_length=max(max_length,input_ids.size(-1))
# 			input_tensors.append(input_ids)
	
# 	# import pdb
# 	# pdb.set_trace()
# 	# Concatenate inputs as a batch
# 	input_ids = torch.zeros((len(input_tensors), max_length)).long()
# 	attention_mask = torch.zeros((len(input_tensors), max_length)).long()
# 	for i in range(len(input_tensors)):
# 		input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
# 		attention_mask[i, :input_tensors[i].size(-1)] = 1

# 	# import pdb
# 	# pdb.set_trace()
# 	# Print some examples
# 	if verbose==True:
# 		print('####### example #######')
# 		print(tokenizer.decode(input_ids[0]))
# 		# print(tokenizer.decode(input_ids[1]))
# 		print('####### example #######\n')

# 	input_ids = input_ids.cuda()
# 	attention_mask = attention_mask.cuda()
# 	assert len(input_tensors) > 0

# 	# Maximum generate content length
# 	max_length = max_gen_length

# 	start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
# 	# start_mask=tokenizer._convert_token_to_id('<PAD>')
# 	ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
# 	ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

# 	# decoder_input_ids: decoder inputs for next regressive generation
# 	# ll: log likelihood
# 	# output_id: which part of generated contents we are at
# 	# output: generated content so far
# 	# last_length (deprecated): how long we have generated for this part
# 	# import pdb
# 	# pdb.set_trace()
# 	batch_size=1
# 	turn =input_ids.size(0)//batch_size
# 	model.eval()
# 	for t in range(turn):
# 		start=t*batch_size
# 		end=min((t+1)*batch_size,input_ids.size(0))
# 		current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 0, 'output': [], 'last_length': -1}]
# 		for i in tqdm(range(max_length - 2)):
# 			new_current_output = []
# 			for item in current_output:
# 				if item['output_id'] > target_numbers[t]:
# 					# Enough contents
# 					new_current_output.append(item)
# 					continue
# 				decoder_input_ids = item['decoder_input_ids']
# 				with torch.no_grad():
# 					logits=model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.cuda()[start:end])[0]
# 					logits=logits[0]
# 				# import pdb
# 				# pdb.set_trace()
# 				log_denominator = torch.logsumexp(logits[i], -1).item()
# 				ids=list(range(model.config.vocab_size))
# 				ids.sort(key=lambda x: logits[i][x].item(),reverse=True)
# 				ids=ids[:beam+3]
# 				for word_id in ids:
# 					output_id=item['output_id']

# 					if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
# 						# Finish one part if word=='<extra_id_%d>' or '</s>'
# 						if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
# 							check = False
# 						else:
# 							check = True
# 						output_id += 1
# 						last_length = 0
# 					else:
# 						last_length = item['last_length'] + 1
# 						check = True

# 					output_text = item['output'] + [word_id]
# 					ll = item['ll'] + logits[i][word_id] - log_denominator
# 					new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
# 					new_decoder_input_ids[:] = decoder_input_ids
# 					new_decoder_input_ids[..., i + 1] = word_id

# 					# Forbid single space token, "....", and ".........."
# 					if word_id in [3, 19794, 22354]:
# 						check = False

# 					# Forbid continuous "."
# 					if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
# 						check = False

# 					if check:
# 						# Add new results to beam search pool
# 						new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
# 						new_current_output.append(new_item)

# 			if len(new_current_output) == 0:
# 				break

# 			new_current_output.sort(key=lambda x: x['ll'], reverse=True)
# 			new_current_output = new_current_output[:beam]
# 			current_output = new_current_output

# 		result = []
# 		for item in current_output:
# 			result.append(tokenizer.decode(item))
# 		# if verbose==True:
# 		# 	print("####### generated results #######")
# 		# 	for item in current_output:
# 		# 		generate_text = ''
# 		# 		for token in item['output']:
# 		# 			generate_text += tokenizer._convert_id_to_token(token)
# 		# 		print('--------------')
# 		# 		print('score:', item['ll'].item())
# 		# 		print('generated ids', item['output'])
# 		# 		print('generated text', generate_text)
# 		# 		result.append(generate_text)
# 		# 	print("####### generated results #######\n")

# 	return result

import time
def t5_generate(strings_to_be_generated, model, tokenizer, target_number, choice_type='beam',beam=10, label=None, length_limit=None, max_gen_length=20, truncate='tail', verbose=False):
	"""
	choice_type: ['beam','max','prob']
	Generate templates based on given inputs
	label: Only use instances with this label (deprecated)
	length_limit: At least generate content as long as length_limit (deprecated)
	"""
	end_token = tokenizer._convert_token_to_id('</s>')
	pad_token = tokenizer._convert_token_to_id('<pad>')
	start_mask_token=tokenizer._convert_token_to_id('<extra_id_99>')
	end_mask_token=tokenizer._convert_token_to_id('<extra_id_0>')
	max_length = max_gen_length
	start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
	ori_decoder_input_ids = torch.zeros((1, max_length)).long()
	ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id
	pred_blanks=[];pred_texts=[]
	model.eval()
	for s in strings_to_be_generated:
		target_number=s.count('<extra_id_')
		inputs=tokenizer(s,return_tensors='pt')
		# import pdb
		# pdb.set_trace()
		current_output=[{'decoder_input_ids':ori_decoder_input_ids,'ll':0,'output_id':0,'output':[],'last_length':-1}]
		for i in tqdm(range(max_length-2)):
			# print(i)
			new_current_output=[]
			total_decoder_input_ids=[]
			my_current_output=[]
			for item in current_output:
				# sess1=time.time()
				# print('sesssssss',target_number,item['output_id'])
				if item['output_id']>target_number:
					new_current_output.append(item)
					continue
				my_current_output.append(item)
				decoder_input_ids=item['decoder_input_ids']
				total_decoder_input_ids.append(decoder_input_ids)
			sess1=time.time()
			if len(my_current_output)!=0:
				with torch.no_grad():
					input_ids=inputs.input_ids
					attention_mask=inputs.attention_mask
					# print('sess1',input_ids[0].shape)
					# print('sess2',total_decoder_input_ids)
					# print('sess3',attention_mask[0].shape)
					# print('sess4',total_decoder_input_ids[0].shape)
					total_logits=model(input_ids=input_ids.expand(len(total_decoder_input_ids),input_ids.shape[1]).cuda(), \
						attention_mask=attention_mask.expand(len(total_decoder_input_ids),attention_mask.shape[1]).cuda(), \
						decoder_input_ids=torch.cat(total_decoder_input_ids).cuda())[0]
			sess2=time.time()
			for (item_id, item) in enumerate(my_current_output):
				# sess4=time.time()
				# import pdb
				# pdb.set_trace()
				logits=total_logits[item_id]
				log_denominator = torch.logsumexp(logits[i], -1).item()
				_, ids = torch.topk(logits[i], k=beam+3, dim=-1, largest=True)
				# sess5=time.time()
				for word_id in ids:
					output_id=item['output_id']
					if (word_id>=start_mask_token and word_id<=end_mask_token) or word_id==end_token or word_id==pad_token:
					# if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
						# Finish one part if word=='<extra_id_%d>' or '</s>'
						if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
							check = False
						else:
							check = True
						output_id += 1
						last_length = 0
					else:
						last_length = item['last_length'] + 1
						check = True
					output_text = item['output'] + [word_id]
					ll = item['ll'] + logits[i][word_id] - log_denominator
					########################## zj ##############################
					new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
					new_decoder_input_ids[:] = decoder_input_ids
					new_decoder_input_ids[..., i + 1] = word_id
					# Forbid single space token, "....", and ".........."
					if word_id in [3, 19794, 22354]:
						check = False
					# Forbid continuous "."
					if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
						check = False
					if check:
						# Add new results to beam search pool
						new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
						new_current_output.append(new_item)
				# sess6=time.time()
				# print('sess5-sess4',sess5-sess4)
				# print('sess6-sess5',sess6-sess5)
			# sess3=time.time()
			# print('sess2-sess1',sess2-sess1)
			# print('sess3-sess2',sess3-sess2)
			if len(new_current_output) == 0:
				break
			new_current_output.sort(key=lambda x: x['ll'], reverse=True)
			new_current_output = new_current_output[:beam]
			current_output = new_current_output

		######################################### postprocess results ############################################
		pred_text=[];result = []
		for item in current_output:
			result.append([])
			blanks=[]
			# print(item['output'])
			for token_id in item['output']:
				token_id=token_id.item()
				if (token_id>=start_mask_token and token_id<=end_mask_token) or token_id==end_token or token_id==pad_token:
					blanks.append([])
				else:
					blanks[-1].append(token_id)
			for blank in blanks:
				result[-1].append(tokenizer.decode(blank))

			current_blank=0;output_tokens=[]
			for token in inputs.input_ids[0]:
				token=token.item()
				if token>=start_mask_token and token<=end_mask_token:
					if current_blank<len(blanks):
						output_tokens+=blanks[current_blank]
					current_blank+=1
				else:
					if token not in [pad_token,end_token]:
						output_tokens.append(token)
			pred_text.append(tokenizer.decode(output_tokens))
		pred_texts.append(pred_text)
		pred_blanks.append(result)
	return pred_texts,pred_blanks

def generate_t5_blanks(strings_to_be_generated,model,tokenizer,beam=10,gen_type='default',max_gen_length=100,num_return_sequences=None,verbose=False):
	model.eval()
	end_token = tokenizer._convert_token_to_id('</s>')
	pad_token = tokenizer._convert_token_to_id('<pad>')
	start_mask_token=tokenizer._convert_token_to_id('<extra_id_99>')
	end_mask_token=tokenizer._convert_token_to_id('<extra_id_0>')
	def postprocess_texts(string_to_be_generated_tokens,generated_text_tokens):
		text=[token for token in generated_text_tokens if token not in [end_token,pad_token]]
		blanks=[[]]
		for token in text:
			if token>=start_mask_token and token<=end_mask_token:
				if len(blanks[-1])!=0:
					blanks.append([])
			else:
				blanks[-1].append(token)
		# import pdb
		# pdb.set_trace()
		output_tokens=[]
		current_blank=0
		for token in string_to_be_generated_tokens[0]:
			if token>=start_mask_token and token<=end_mask_token:
				if current_blank < len(blanks):
					output_tokens+=blanks[current_blank]
				current_blank+=1
			else:
				if token not in [pad_token]:
					output_tokens.append(token)
		pred_text=tokenizer.decode(output_tokens)
		pred_blank=[tokenizer.decode(x) for x in blanks]
		return pred_text,pred_blank

	total_predictions=[];pred_blanks=[]
	for string_to_be_generated in tqdm(strings_to_be_generated):
		if verbose==True:
			print('>> input_sentences:\n',string_to_be_generated)
		if gen_type=='default':
			input_tokens=tokenizer(''.join(string_to_be_generated),return_tensors='pt')
			if beam==1: 
				if num_return_sequences is None:
					num_return_sequences=1
				do_sample=True
			else: 
				if num_return_sequences is None:
					num_return_sequences=1
				do_sample=False
			# else: num_return_sequences=int(np.ceil(beam//2));do_sample=False
			predected_results=model.generate(input_ids=input_tokens.input_ids.cuda(),attention_mask=input_tokens.attention_mask.cuda(),
			max_length=max_gen_length,num_beams=beam,num_return_sequences=num_return_sequences,do_sample=do_sample,
			repetition_penalty=2.5,length_penalty=1.0,early_stopping=True)
			for (i,generated_text_tokens) in enumerate(predected_results):
				# print('>> my_prediction_{}:\n'.format(i),tokenizer.decode(generated_text_tokens))
				if verbose==True:
					print('>> my_prediction_{}:\n'.format(i))
				pred_text,pred_blank=postprocess_texts(input_tokens.input_ids.detach().cpu().numpy(),generated_text_tokens.detach().cpu().numpy())
				total_predictions.append(pred_text)
				pred_blanks.append(pred_blank)
			# print('')
		else:
			pred_text,pred_blank=t5_generate([string_to_be_generated],model, tokenizer, target_number=2, beam=beam, label=None,max_gen_length=max_gen_length,truncate='head')[:beam//2]
			total_predictions+=pred_text[0]
			pred_blanks+=pred_blank[0]
	return total_predictions, pred_blanks

