task_name=$1
device=$2
LR_CANDIDATES=$3 # 1e-5 5e-6
MAX_STEPS=$4  # 5000 2500
# every_eval_ratios=$5 # 0.02 0.04

model_type="deberta"
few_shot_setting="dev32_split"
dataset_name="superglue"
method="sequence_classifier"
arch_method="default"
data_dir='/workspace/yanan/few-shot/FewGLUE_dev32/'
save_dir="0811_cls_checkpoints/${few_shot_setting}/${model_type}_${task_name}_${method}_model"


if [ $model_type = "albert" ]; then
  model_name_or_path="/workspace/zhoujing/huggingface_models/albert-xxlarge-v2"
  TRAIN_BATCH_SIZE_CANDIDATES="8"

elif [ $model_type = "deberta" ]; then
  model_name_or_path="/workspace/zhoujing/huggingface_models/deberta-v2-xxlarge"
  TRAIN_BATCH_SIZE_CANDIDATES="2"
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

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB


elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96

elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE_CANDIDATES="1"


elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16

else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi

WARMUP_RATIO="0.0"
SAMPLER_SEED="42"
SEED="42"
cv_k="4"
every_eval_ratios="1"

for MAX_STEP in $MAX_STEPS
do
  for TOTAL_TRAIN_BATCH in $TOTAL_BATCH_SIZE_CANDIDATES
  do
    for TRAIN_BATCH_SIZE in $TRAIN_BATCH_SIZE_CANDIDATES
    do
      for LR in $LR_CANDIDATES
      do
        for every_eval_ratio in $every_eval_ratios
        do
        ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
        HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_${TOTAL_TRAIN_BATCH}_${TRAIN_BATCH_SIZE}_${LR}_${every_eval_ratio}_${cv_k}
        OUTPUT_DIR=$save_dir/${HYPER_PARAMS}

        CUDA_VISIBLE_DEVICES=$device nohup python3 cli.py \
          --method $method \
          --arch_method $arch_method \
          --data_dir $DATA_DIR \
          --pattern_ids 0 \
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
          --few_shot_setting $few_shot_setting \
          --every_eval_ratio $every_eval_ratio \
          --cv_k $cv_k \
          --overwrite_output_dir \
          --fix_deberta >myout_${few_shot_setting}_${method}_${task_name}.file 2>&1 &
          wait
        done
      done
    done
  done
done



# BoolQ:
# nohup bash search_cls_devsplit.sh boolq 0 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh boolq 1 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh boolq 2 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh boolq 3 1e-5 5000 >myout.file 2>&1 &

# nohup bash search_cls_devsplit.sh rte 4 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh rte 5 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh rte 6 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh rte 7 1e-5 5000 >myout.file 2>&1 &


# WiC:
# nohup bash search_cls_devsplit.sh wic 0 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wic 1 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wic 2 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wic 3 1e-5 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 4 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 5 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 6 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 7 1e-5 5000 >myout.file 2>&1 &



# MultiRC:
# nohup bash search_cls_devsplit.sh multirc 0 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh multirc 1 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh multirc 2 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh multirc 3 1e-5 5000 >myout.file 2>&1 &

# nohup bash search_cls_devsplit.sh wsc 4 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wsc 5 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wsc 6 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wsc 7 1e-5 5000 >myout.file 2>&1 &


# copa:
# nohup bash search_cls_devsplit.sh copa 0 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh copa 1 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh copa 2 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh copa 3 1e-5 5000 >myout.file 2>&1 &

###########################################################################
# nohup bash search_cls_devsplit.sh boolq 0 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh boolq 1 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh rte 2 5e-6 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh rte 3 1e-5 2500 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wic 4 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh wic 5 1e-5 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 6 5e-6 5000 >myout.file 2>&1 &
# nohup bash search_cls_devsplit.sh cb 7 1e-5 5000 >myout.file 2>&1 &

