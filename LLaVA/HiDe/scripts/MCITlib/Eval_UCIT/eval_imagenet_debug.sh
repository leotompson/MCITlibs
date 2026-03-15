#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="ImageNet-R"
# 虽然保留了读取，但在下方强制指定单卡
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
TEXT_TOWER=$(read_config "$TRAIN_CONFIG" text_tower)
NUM_TASK=$(read_config "$TRAIN_CONFIG" num_task)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_config "$DATA_CONFIG" test_folder)
RESULT_PATH=$(read_config "$TRAIN_CONFIG" result_path)

# --- 修改部分：强制指定使用第 0 号 GPU ---
export CUDA_VISIBLE_DEVICES="0" 
CHUNKS=1
IDX=0
# ---------------------------------------

RESULT_DIR="$RESULT_PATH/$TASK"
# 确保目录存在
mkdir -p "$RESULT_DIR/$STAGE"

echo "正在使用单 GPU 进行推理..."

# 去掉了 for 循环和后台运行符号 &
# 直接运行单个 Python 进程
python -m llava.eval.CoIN.model_others \
    --model-path $MODELPATH \
    --model-base $MODELBASE \
    --question-file $DATA_PATH \
    --image-folder $IMAGE \
    --text-tower $TEXT_TOWER \
    --num-task $NUM_TASK \
    --answers-file $RESULT_DIR/$STAGE/single_gpu_result.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1

# 因为只有单卡，不需要 wait 和合并 (merge) 文件的循环
output_file=$RESULT_DIR/$STAGE/merge.jsonl
mv $RESULT_DIR/$STAGE/single_gpu_result.jsonl "$output_file"

echo "推理完成，开始评估..."

python -m llava.eval.CoIN.eval_deepseek_r1 \
    --annotation-file $DATA_PATH \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE