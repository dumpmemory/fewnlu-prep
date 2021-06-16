import torch
from genaug import confidence_filter
device='cuda:0'
task_name='rte'
new_examples=torch.load(os.path.join('/data/zhoujing/NLP/ptuning/results/augmented_examples','t5_0.5_rte'))
pattern_maps={'rte':1,'boolq':4,'cb':2}
pattern_id=pattern_maps[task_name]
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/data/zhoujing/NLP/baseline/PET/PET_2.0.0_myself/results/several_seed_ensemble/methods/{}_32_albert_model/seed_0/p{}-i0'.format(task_name,pattern_id))
eval_config = pet.EvalConfig(device=device, n_gpu=1, metrics='acc', per_gpu_eval_batch_size=8, decoding_strategy='default', priming=False)
example_num=len(train_data)
examples=[]
for i in range(int(np.ceil(len(new_examples)/example_num))):
	examples.append(new_examples[i*example_num:(i+1)*example_num])
# print(len(examples),len(examples[0]))
# print(examples[0][0])
new_examples=myfilter.recover_labels(myfilter.wrapper,pattern_id,examples,eval_config,recover_type=args.search_type.split('_filter_')[1])
print(new_examples)
myfilter.del_finetuned_model()