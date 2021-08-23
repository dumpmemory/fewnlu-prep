task_name=$1
device=$2
LR_CANDIDATES=$3
MAX_STEPS=$4
PATTERN_IDS=$5
cv_k=$6


model_type="deberta"
few_shot_setting="mdl"
dataset_name="superglue"
method="pet"
arch_method="default"
# data_dir='/workspace/yanan/zyn/few-shot/FewGLUE_dev32/'
data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
save_dir="0718_pet_checkpoints/${few_shot_setting}/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
#   model_name_or_path="albert-xxlarge-v2"
  # model_name_or_path="albert-xxlarge-v2"
  model_name_or_path="/workspace/zhoujing/huggingface_models/albert-xxlarge-v2"
  # TRAIN_BATCH_SIZE_CANDIDATES="8 4 2 1"
  LR_CANDIDATES="1e-5 2e-5"

elif [ $model_type = "deberta" ]; then
  # model_name_or_path="microsoft/deberta-v2-xxlarge"
  model_name_or_path="/workspace/zhoujing/huggingface_models/deberta-v2-xxlarge"
  # TRAIN_BATCH_SIZE_CANDIDATES="2 1" #"2 1"
  TRAIN_BATCH_SIZE_CANDIDATES="2"
  # LR_CANDIDATES="5e-6 1e-5"
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
TOTAL_BATCH_SIZE_CANDIDATES="16"
EVAL_BATCH_SIZE=32
DATA_ROOT=$data_dir
TASK=$task_name


if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC
  # PATTERN_IDS="0 1 2"

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE
  # PATTERN_IDS="0 1 2 3 4 5"

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB
  # PATTERN_IDS="0 1 2 3 4 5"

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1
  # PATTERN_IDS="0 1 2"
  max_num_lbl_tok=20

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ
  # PATTERN_IDS="0 1 2 3 4 5"

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1
  # PATTERN_IDS="0 1"
  max_num_lbl_tok=20


elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  # PATTERN_IDS="0 1 2"
  TRAIN_BATCH_SIZE_CANDIDATES="1"


elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  # PATTERN_IDS="0"

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


# MAX_STEPS="250 500"
WARMUP_RATIO="0.0"
SAMPLER_SEED="42"
SEED="42"
every_eval_ratios="0.02 0.04"


for MAX_STEP in $MAX_STEPS
do
  for TOTAL_TRAIN_BATCH in $TOTAL_BATCH_SIZE_CANDIDATES
  do
    for TRAIN_BATCH_SIZE in $TRAIN_BATCH_SIZE_CANDIDATES
    do
      for LR in $LR_CANDIDATES
      do
        for PATTERN in $PATTERN_IDS
        do
        for every_eval_ratio in $every_eval_ratios
        do
        ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
        HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_${TOTAL_TRAIN_BATCH}_${TRAIN_BATCH_SIZE}_${LR}_${PATTERN}_${every_eval_ratio}_${cv_k}
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
          --seed $SEED \
          --warmup_step_ratio $WARMUP_RATIO \
          --learning_rate $LR \
          --repetitions 1 \
          --use_cloze \
          --few_shot_setting $few_shot_setting \
          --every_eval_ratio $every_eval_ratio \
          --cv_k $cv_k \
          --fix_deberta >myout_${few_shot_setting}_${method}_${task_name}.file 2>&1 &
          wait
        done
        done
      done
    done
  done
done


# boolq
# 256_250_16_2_1e-5_5_0.04_8
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 250 5 8 >myout.file 2>&1 &

# rte 
# 256_500_16_2_5e-6_4_0.02_8
# 256_500_16_2_5e-6_4_0.04_8
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 500 4 8 >myout.file 2>&1 &

# BoolQ
# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 250 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 6 5e-6 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 5e-6 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 6 5e-6 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 5e-6 500 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 6 1e-5 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 1e-5 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 1e-5 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 1e-5 250 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 4 1e-5 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 1e-5 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 500 5 16 >myout.file 2>&1 &



# RTE
# nohup bash search_pet_mdl_k.sh rte 5 5e-6 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 5e-6 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 5e-6 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 5e-6 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 5e-6 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 5e-6 250 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 2 5e-6 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 5e-6 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 5e-6 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 5e-6 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 7 5e-6 500 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 3 1e-5 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 250 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 7 1e-5 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 7 1e-5 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 1e-5 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 1e-5 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 500 5 16 >myout.file 2>&1 &

# copa 
# nohup bash search_pet_mdl_k.sh copa 4 5e-6 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 5 5e-6 250 1 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 7 5e-6 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 3 5e-6 500 1 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 4 1e-5 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 5 1e-5 250 1 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 0 1e-5 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 5 1e-5 500 1 16 >myout.file 2>&1 &
# todo 
# CB
# nohup bash search_pet_mdl_k.sh cb 3 5e-6 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 5e-6 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 7 5e-6 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 5e-6 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 5e-6 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 250 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 3 5e-6 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 5e-6 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 5e-6 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 5e-6 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 5e-6 500 5 16 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 5 1e-5 250 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 1e-5 250 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 1e-5 250 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 1e-5 250 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 1e-5 250 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 3 1e-5 250 5 16 >myout.file 2>&1 &


# nohup bash search_pet_mdl_k.sh cb 4 1e-5 500 0 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 1 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 1e-5 500 2 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 7 1e-5 500 3 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 1e-5 500 4 16 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 5 16 >myout.file 2>&1 &





# BoolQ
# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 5e-6 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 6 5e-6 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 250 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 5e-6 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 5e-6 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 5e-6 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 500 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 1e-5 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 1e-5 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 1e-5 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 250 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 4 1e-5 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 1e-5 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 6 1e-5 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 1e-5 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 500 5 2 >myout.file 2>&1 &



# RTE
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 5e-6 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 5e-6 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 5e-6 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 5e-6 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 5e-6 250 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 2 5e-6 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 5e-6 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 7 5e-6 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 5e-6 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 5e-6 500 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 3 1e-5 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 1e-5 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 1e-5 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 250 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 3 1e-5 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 1e-5 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 1e-5 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 500 5 2 >myout.file 2>&1 &


# copa
# nohup bash search_pet_mdl_k.sh copa 3 5e-6 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 4 5e-6 250 1 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 5 5e-6 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 1 5e-6 500 1 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 0 1e-5 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 1 1e-5 250 1 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 2 1e-5 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 6 1e-5 500 1 2 >myout.file 2>&1 &

# CB
# nohup bash search_pet_mdl_k.sh cb 0 5e-6 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 3 5e-6 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 5e-6 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 5e-6 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 5e-6 250 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 0 5e-6 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 5e-6 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 3 5e-6 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 5e-6 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 5e-6 500 5 2 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 6 1e-5 250 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 7 1e-5 250 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 1e-5 250 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 1e-5 250 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 1e-5 250 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 3 1e-5 250 5 2 >myout.file 2>&1 &


# nohup bash search_pet_mdl_k.sh cb 4 1e-5 500 0 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 1 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 1e-5 500 2 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 1e-5 500 3 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 4 2 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 5 2 >myout.file 2>&1 &




# BoolQ
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 5e-6 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 5e-6 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 5e-6 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 250 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 7 5e-6 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 5e-6 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 5e-6 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 5e-6 500 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 5 1e-5 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 4 1e-5 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 0 1e-5 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 1 1e-5 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 250 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 2 1e-5 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 1e-5 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 5 1e-5 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 3 1e-5 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh boolq 7 1e-5 500 5 8 >myout.file 2>&1 &



# RTE
# nohup bash search_pet_mdl_k.sh rte 6 5e-6 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 5e-6 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 5e-6 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 5e-6 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 6 5e-6 250 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 7 5e-6 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 5e-6 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 5e-6 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 5e-6 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 5e-6 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 0 5e-6 500 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 1 1e-5 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 3 1e-5 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 1e-5 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 7 1e-5 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 250 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh rte 2 1e-5 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 1e-5 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 1 1e-5 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 2 1e-5 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 4 1e-5 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh rte 5 1e-5 500 5 8 >myout.file 2>&1 &

# copa
# nohup bash search_pet_mdl_k.sh copa 6 5e-6 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 7 5e-6 250 1 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 0 5e-6 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 1 5e-6 500 1 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 6 1e-5 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 1 1e-5 250 1 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh copa 2 1e-5 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh copa 3 1e-5 500 1 8 >myout.file 2>&1 &

# todo 
# CB
# nohup bash search_pet_mdl_k.sh cb 3 5e-6 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 5e-6 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 7 5e-6 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 5e-6 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 5e-6 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 250 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 3 5e-6 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 5e-6 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 5e-6 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 5e-6 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 5e-6 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 4 5e-6 500 5 8 >myout.file 2>&1 &

# nohup bash search_pet_mdl_k.sh cb 5 1e-5 250 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 1e-5 250 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 0 1e-5 250 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 1 1e-5 250 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 1e-5 250 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 3 1e-5 250 5 8 >myout.file 2>&1 &


# nohup bash search_pet_mdl_k.sh cb 4 1e-5 500 0 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 1 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 6 1e-5 500 2 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 7 1e-5 500 3 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 2 1e-5 500 4 8 >myout.file 2>&1 &
# nohup bash search_pet_mdl_k.sh cb 5 1e-5 500 5 8 >myout.file 2>&1 &