task_name=$1
device=$2
model_type=$3
few_shot_setting=$4

dataset_name="superglue"
method="pet"
# arch_method="default"
arch_method="ipet"
# data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
data_dir='/workspace/yanan/zyn/few-shot/FewGLUE_dev32/'
# data_dir='/workspace/zhoujing/FewGLUE_dev32/'
save_dir="pet_checkpoints/${model_type}_${task_name}_${method}_model"
# save_dir="test/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
#   model_name_or_path="albert-xxlarge-v2"
  model_name_or_path="/workspace/zhoujing/data/checkpoints/albert-xxlarge-v2"
  TRAIN_BATCH_SIZE_CANDIDATES="8 4 2 1"
  LR_CANDIDATES="1e-5 2e-5"

elif [ $model_type = "deberta" ]; then
  model_name_or_path="/workspace/zhoujing/huggingface_models/deberta-v2-xxlarge"
  TRAIN_BATCH_SIZE_CANDIDATES="2 1"
  LR_CANDIDATES="1e-5 5e-6"
fi


echo Running with the following parameters:
echo ------------------------------------
echo DATASET_NAME           = "$dataset_name"
echo TASK_NAME              = "$task_name"
echo METHOD                 = "$method"
echo ARCH_METHOD            = "$arch_method"
echo DEVICE                 = "$device"
echo MODEL_TYPE             = "$model_type"
echo MODEL_NAME_OR_PATH     = "$model_name_or_path"
echo DATA_ROOT              = "$data_dir"
echo SAVE_DIR               = "$save_dir"
echo ------------------------------------


SEQ_LENGTH=256
EVAL_BATCH_SIZE=32
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
  PATTERN_IDS="0 1 2 3 4"

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


MAX_STEP=50
TOTAL_BATCH_SIZE=16
TRAIN_BATCH_SIZE=2
every_eval_ratio=0.02
LR=1e-5
SAMPLER_SEED="10 20 30"
PATTERN_IDS="1 2"
ACCU=$((${TOTAL_BATCH_SIZE}/${TRAIN_BATCH_SIZE}))
HYPER_PARAMS=${MAX_STEP}_${TRAIN_BATCH_SIZE}_${ACCU}_${LR}_${SEQ_LENGTH}
OUTPUT_DIR=$save_dir/${HYPER_PARAMS}

CUDA_VISIBLE_DEVICES=$device python3 cli.py \
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
  --sampler_seeds $SAMPLER_SEED \
  --learning_rate $LR \
  --repetitions 2 \
  --use_cloze \
  --fix_deberta \
  --few_shot_setting $few_shot_setting \
  --every_eval_ratio $every_eval_ratio \
  --cv_k 2 \
  --use_brother_fold_logits \
  --overwrite_output_dir

# bash zj_run_superglue_pet.sh wic 7 deberta dev32_split
