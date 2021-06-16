################################## prepare augmented examples ############################################
################################## prepare model ############################################
from genaug import gen_aug_T5
from genaug import gen_aug_pattern
import imp
imp.reload(gen_aug_pattern)
t5_args=gen_aug_T5.init_t5_args()
tokenizer,model=gen_aug_T5.init_model()


################################## read dataset #############################################
import os
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
data_dir='/data/zhoujing/NLP/data/FewGLUE'
task_name='rte'
train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]) , TRAIN_SET,num_examples=-1)
unlabeled_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), UNLABELED_SET, num_examples=-1)
test_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), DEV_SET, num_examples=-1)

augmented_examples=gen_aug_pattern.aug_general(task_name,train_examples,gen_aug_T5.generate_t5_blanks,model,tokenizer,aug_num=10,mask_ratio=0.5,save_path='results/augmented_examples',aug_type='default')


import os
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
data_dir='/data/zhoujing/NLP/data/FewGLUE'
task_name='boolq'
train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]) , TRAIN_SET,num_examples=-1)
unlabeled_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), UNLABELED_SET, num_examples=-1)
test_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), DEV_SET, num_examples=-1)

augmented_examples=gen_aug_pattern.aug_general(task_name,train_examples,gen_aug_T5.generate_t5_blanks,model,tokenizer,aug_num=10,mask_ratio=0.5,save_path='results/augmented_examples',aug_type='default')


import os
from methods.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
NAME={'boolq':'BoolQ','cb':'CB','copa':'COPA','rte':'RTE','wic':'WiC','wsc':'WSC','multirc':'MultiRC','record':'ReCoRD'}
data_dir='/data/zhoujing/NLP/data/FewGLUE'
task_name='cb'
train_examples=load_examples(task_name,os.path.join(data_dir,NAME[task_name]) , TRAIN_SET,num_examples=-1)
unlabeled_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), UNLABELED_SET, num_examples=-1)
test_examples = load_examples(task_name, os.path.join(data_dir,NAME[task_name]), DEV_SET, num_examples=-1)

augmented_examples=gen_aug_pattern.aug_general(task_name,train_examples,gen_aug_T5.generate_t5_blanks,model,tokenizer,aug_num=10,mask_ratio=0.5,save_path='results/augmented_examples',aug_type='default')


################################## prepare augmented examples ############################################





















##################################### filter ##########################################


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
new_examples=myfilter.recover_labels(myfilter.wrapper,pattern_id,examples,eval_config,recover_type="max_eachla")
# print(new_examples)
myfilter.del_finetuned_model()