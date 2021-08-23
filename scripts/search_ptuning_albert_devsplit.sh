task_name=$1
device=$2
MAX_STEPS=$3 # 250 500
every_eval_ratios=$4 # 0.02+0.04
WARMUP_RATIO=$5

LR_CANDIDATES=$6 #"1e-5 2e-5 3e-5"

# space: (250+500)*(0.02+0.04)*(5e-6 + 1e-5)*1*(mlp+lstm)*(0.0+0.1)=32

model_type="albert"
# model_type="deberta"
few_shot_setting="dev32_split"
dataset_name="superglue"
method="ptuning"
arch_method="default"
data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
save_dir="0808_ptuning_checkpoints/${few_shot_setting}/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
  # model_name_or_path="albert-xxlarge-v2"
  # model_name_or_path="albert-xxlarge-v2"
  model_name_or_path="/workspace/zhoujing/huggingface_models/albert-xxlarge-v2"
  # TRAIN_BATCH_SIZE_CANDIDATES="8 4 2 1"
  TRAIN_BATCH_SIZE_CANDIDATES="8"
#   LR_CANDIDATES="1e-5 2e-5"

elif [ $model_type = "deberta" ]; then
  # model_name_or_path="microsoft/deberta-v2-xxlarge"
  model_name_or_path="/workspace/zhoujing/huggingface_models/deberta-v2-xxlarge"
  # TRAIN_BATCH_SIZE_CANDIDATES="2 1" #"2 1"
  TRAIN_BATCH_SIZE_CANDIDATES="2"
  # LR_CANDIDATES="5e-6 6e-6 4e-6"
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
TOTAL_TRAIN_BATCH=16
EVAL_BATCH_SIZE=32
DATA_ROOT=$data_dir
TASK=$task_name
#max_num_lbl_tok=1
PATTERN_IDS="1"
TRAIN_BATCH_SIZE=2

if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC
  #PATTERN_IDS="1 2 3 4 5 6"

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTEs
  #PATTERN_IDS="1 2 3 4 5 6"

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB
  #PATTERN_IDS="1 2 3 4 5 6"

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128
  EVAL_BATCH_SIZE=1
  #PATTERN_IDS="1 2 3 4 5 6"
  #max_num_lbl_tok=20

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ
  #PATTERN_IDS="1 2 3 4 5 6"

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96
  EVAL_BATCH_SIZE=1
  #PATTERN_IDS="1 2 3 4 5 6"
  #max_num_lbl_tok=20


elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  #PATTERN_IDS="1 2 3 4"
  TRAIN_BATCH_SIZE=1

elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  #PATTERN_IDS="0"
  TRAIN_BATCH_SIZE=1

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


# MAX_STEPS="250 500"
# WARMUP_RATIO="0.0 0.1"
SAMPLER_SEED="42"
SEED="42"
# every_eval_ratios="0.02 0.04"
HEAD_TYPES="lstm mlp"
cv_k="4"

EMB_LRS="1e-4 2e-4"
for MAX_STEP in $MAX_STEPS # 2
do
  for warmup_ratio in $WARMUP_RATIO # 2
  do
    for every_eval_ratio in $every_eval_ratios # 2
    do
      for LR in $LR_CANDIDATES # 3
      do
        for prompt_encoder_head_type in $HEAD_TYPES # 2
        do
        for EMB_LR in $EMB_LRS
        do
        ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
        HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_16_2_${LR}_${EMB_LR}_1_${every_eval_ratio}_${warmup_ratio}_${prompt_encoder_head_type}
        OUTPUT_DIR=$save_dir/${HYPER_PARAMS}
        echo $LR, $EMB_LR
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
          --seed $SEED \
          --warmup_step_ratio $warmup_ratio \
          --learning_rate $LR \
          --repetitions 1 \
          --embedding_learning_rate $EMB_LR \
          --use_continuous_prompt \
          --prompt_encoder_head_type $prompt_encoder_head_type \
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

# source /workspace/yanan/anaconda3/etc/profile.d/conda.sh
# conda activate deberta-pet-env

# nohup bash search_ptuning_albert_devsplit.sh wsc 0 250 0.02 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 1 250 0.02 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 2 500 0.02 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 3 500 0.02 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 4 250 0.02 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 5 250 0.02 0.1 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 6 500 0.02 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 7 500 0.02 0.1 5e-6 >myout.file 2>&1 &

# nohup bash search_ptuning_albert_devsplit.sh wsc 5 250 0.04 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 1 250 0.04 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 2 500 0.04 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 3 500 0.04 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 4 250 0.04 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 5 250 0.04 0.1 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 6 500 0.04 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 7 500 0.04 0.1 5e-6 >myout.file 2>&1 &

# nohup bash search_ptuning_albert_devsplit.sh wsc 0 250 0.08 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 1 250 0.08 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 2 500 0.08 0.0 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 3 500 0.08 0.0 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 0 250 0.08 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 1 250 0.08 0.1 5e-6 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 3 500 0.08 0.1 1e-5 >myout.file 2>&1 &
# nohup bash search_ptuning_albert_devsplit.sh wsc 4 500 0.08 0.1 5e-6 >myout.file 2>&1 &



