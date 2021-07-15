task_name=$1
device=$2
model_type=$3

dataset_name="superglue"
method="pet"
arch_method='noisy_student'
data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
save_dir="ns_checkpoints/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
  model_name_or_path="albert-xxlarge-v2"
  TRAIN_BATCH_SIZE_CANDIDATES="8 4 2 1"
  LR_CANDIDATES="1e-5 2e-5"

elif [ $model_type = "deberta" ]; then
  model_name_or_path="microsoft/deberta-v2-xxlarge"
  TRAIN_BATCH_SIZE_CANDIDATES="2 1"
  LR_CANDIDATES="1e-5 5e-6"
fi


echo Running with the following parameters:
echo ------------------------------------
echo DATASET_NAME           = "$dataset_name"
echo TASK_NAME              = "$task_name"
echo METHOD                 = "$method"
echo DEVICE                 = "$device"
echo MODEL_TYPE             = "$model_type"
echo MODEL_NAME_OR_PATH     = "$model_name_or_path"
echo DATA_ROOT              = "$data_dir"
echo SAVE_DIR               = "$save_dir"
echo ------------------------------------


SEQ_LENGTH=256
EVAL_BATCH_SIZE=32
PATTERN_IDS="0 1"
DATA_ROOT=$data_dir
TASK=$task_name
if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC
  PATTERN_IDS="0 1 2"

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE
  PATTERN_IDS="0 1 2 3 4"

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB
  # PATTERN_IDS="0 1 2 3"
  PATTERN_IDS="0 1"

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1
  PATTERN_IDS="0 1 2"

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ
  PATTERN_IDS="0 1 2 3 4 5"

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1
  PATTERN_IDS="0 1"

elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  PATTERN_IDS="0 1 2"

elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  PATTERN_IDS="0"

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


MAX_STEP=250
TOTAL_BATCH_SIZE=32
TRAIN_BATCH_SIZE=2
LR=1e-5
SAMPLER_SEED=10
ACCU=$((${TOTAL_BATCH_SIZE}/${TRAIN_BATCH_SIZE}))
HYPER_PARAMS=${MAX_STEP}_${TOTAL_BATCH_SIZE}_${TRAIN_BATCH_SIZE}_${ACCU}_${LR}_${PATTERN}_${SAMPLER_SEED}_${SEQ_LENGTH}
OUTPUT_DIR=$save_dir/${HYPER_PARAMS}

CUDA_VISIBLE_DEVICES=$device nohup python3 cli.py \
--method $method \
--arch_method $arch_method \
--data_dir $DATA_DIR \
--pattern_ids $PATTERN_IDS \
--model_type $model_type \
--model_name_or_path $model_name_or_path \
--dataset_name $dataset_name \
--task_name $task_name \
--output_dir $OUTPUT_DIR \
--do_eval \
--do_train \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--gradient_accumulation_steps $ACCU \
--max_seq_length $SEQ_LENGTH \
--max_steps $MAX_STEP \
--sampler_seed $SAMPLER_SEED \
--learning_rate $LR \
--repetitions 1 \
--generations 3 \
--use_cloze \
--overwrite_output_dir >myout_noisy_${task_name}.file 2>&1 &



# CUDA_VISIBLE_DEVICES=$device python3 cli.py \
# --method $method \
# --arch_method $arch_method \
# --data_dir $DATA_DIR \
# --pattern_ids $PATTERN_IDS \
# --model_type $model_type \
# --model_name_or_path $model_name_or_path \
# --dataset_name $dataset_name \
# --task_name $task_name \
# --output_dir $OUTPUT_DIR \
# --do_eval \
# --do_train \
# --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
# --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
# --gradient_accumulation_steps $ACCU \
# --max_seq_length $SEQ_LENGTH \
# --max_steps $MAX_STEP \
# --sampler_seed $SAMPLER_SEED \
# --learning_rate $LR \
# --repetitions 1 \
# --generations 3 \
# --use_cloze \
# --aug_data_dir ${data_dir}CB/train.jsonl \
# --relabel_aug_data



# bash zj_run_superglue_noisystudent.sh cb 1 albert
# bash zj_run_superglue_noisystudent.sh copa 2 albert
# bash zj_run_superglue_noisystudent.sh wic 5 albert
# bash zj_run_superglue_noisystudent.sh rte 6 albert
# bash zj_run_superglue_noisystudent.sh boolq 7 albert
# bash zj_run_superglue_noisystudent.sh wic 7 albert
# bash zj_run_superglue_noisystudent.sh multirc 7 albert