task_name=$1
device=$2
model_type=$3

prompt_encoder_head_type="mlp"
dataset_name="superglue"
method="ptuning"
arch_method="default"
data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
save_dir="ptuning_checkpoints/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
  # model_name_or_path="albert-xxlarge-v2"
  model_name_or_path="/workspace/zhoujing/data/checkpoints/albert-xxlarge-v2"
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
echo PROMPT_ENCODER_HEAD_TYPE = "$prompt_encoder_head_type"
echo ------------------------------------


SEQ_LENGTH=256
EVAL_BATCH_SIZE=32
PATTERN_IDS="1 2 3 4"
DATA_ROOT=$data_dir
TASK=$task_name
if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1

elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16

elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi



MAX_STEPS="250 500 750 1000"
# MAX_STEPS="5"
EMB_LRS="5e-5 1e-4"
WARMUP_RATIO=0.0
TOTAL_BATCH_SIZE=16
TRAIN_BATCH_SIZE=2
LR=1e-5
SAMPLER_SEED="10 20 30"
for MAX_STEP in $MAX_STEPS
do
for LR in $LR_CANDIDATES
do
for embedding_learning_rate in $EMB_LRS
do
    for PATTERN in $PATTERN_IDS
    do
        ACCU=$((${TOTAL_BATCH_SIZE}/${TRAIN_BATCH_SIZE}))
        HYPER_PARAMS=${MAX_STEP}_${WARMUP_RATIO}_${TOTAL_BATCH_SIZE}_${TRAIN_BATCH_SIZE}_${LR}_${embedding_learning_rate}_${PATTERN}_${SEQ_LENGTH}
        OUTPUT_DIR=$save_dir/${HYPER_PARAMS}
        CUDA_VISIBLE_DEVICES=$device nohup python3 cli.py \
        --method $method \
        --arch_method $arch_method \
        --data_dir $DATA_DIR \
        --pattern_ids $PATTERN \
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
        --repetitions 3 \
        --embedding_learning_rate $embedding_learning_rate\
        --use_continuous_prompt \
        --prompt_encoder_head_type $prompt_encoder_head_type \
        --warmup_step_ratio $WARMUP_RATIO \
        --use_cloze \
        --fix_deberta >myout_${task_name}.file 2>&1 &
        wait
done
done
done
done

# nohup bash zj_search_ptuning.sh rte 1 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh boolq 2 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh wic 3 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh wsc 4 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh cb 5 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh copa 6 deberta >nohup.file 2>&1 &
# nohup bash zj_search_ptuning.sh multirc 7 deberta >nohup.file 2>&1 &
