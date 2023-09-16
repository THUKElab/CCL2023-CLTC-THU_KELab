#!/bin/bash

while getopts "g:i:o:v:m:r:d:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    v)
        DIR_VALID=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    d)
        DATASET=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=0

BATCH_SIZE=64

UPDATE_FREQ=8

DIR_MODEL=${DIR_MODEL:-"ccl_pn_bt+ccl_denoise_ensemble_s5e5_6-seed${SEED}"}

DIR_VALID=${DIR_VALID:-"models/seq2edit/preprocess/zho/yaclc_minimal_dev"}

DATASET_VALID="yaclc_minimal_dev"

PRETRAIN_WEIGHTS_DIR="../resources/chinese-struct-bert-large"

VOCAB_PATH="models/seq2edit/preprocess/zho/output_vocabulary_chinese_char_hsk+lang8_5"

#DIR_MODEL_STAGE0=models/seq2edit/exps/zho/stage0/${DIR_MODEL}
DIR_MODEL_STAGE1=models/seq2edit/exps/zho/stage1/${DIR_MODEL}
DIR_MODEL_STAGE2=models/seq2edit/exps/zho/stage2/${DIR_MODEL}

#DIR_INPUT_STAGE0=models/seq2edit/preprocess/zho/pattern_noise/ccl_pn0.1*4+bt6.0*4
DIR_INPUT_STAGE1=models/seq2edit/preprocess/zho/real/ccl_denoise/mediate_ensemble_s5e5_6
DIR_INPUT_STAGE2=models/seq2edit/preprocess/zho/yaclc_dev

#mkdir -p ${DIR_MODEL_STAGE0}
mkdir -p ${DIR_MODEL_STAGE1}
mkdir -p ${DIR_MODEL_STAGE2}
#mkdir -p ${DIR_MODEL_STAGE0}/results
mkdir -p ${DIR_MODEL_STAGE1}/results
mkdir -p ${DIR_MODEL_STAGE2}/results

#cp "../datasets/GEC/CGEC/mucgec/train/MuCGEC_train.src" "models/fairseq/bart/preprocess/zho/real/chinese_hsk+lang8/src.txt"
#cp "../datasets/GEC/CGEC/mucgec/train/MuCGEC_train.tgt" "models/fairseq/bart/preprocess/zho/real/chinese_hsk+lang8/tgt.txt"

#if [ ! -d ${DIR_INPUT_STAGE0}/train.char.label ]; then
#    bash scripts/seq2edit/preprocess.sh  -i ${DIR_INPUT_STAGE0}  -o ${DIR_INPUT_STAGE0}  -v ${DIR_VALID}
#fi

if [ ! -d ${DIR_INPUT_STAGE1}/train.char.label ]; then
    bash scripts/seq2edit/preprocess.sh  -i ${DIR_INPUT_STAGE1}  -o ${DIR_INPUT_STAGE1}  -v ${DIR_VALID}
fi

#if [ ! -d ${DIR_INPUT_STAGE2}/train.char.label ]; then
#    bash scripts/seq2edit/preprocess.sh  -i ${DIR_INPUT_STAGE2}  -o ${DIR_INPUT_STAGE2}  -v ${DIR_VALID}
#fi

## ========================= Seq2Edit stage 0 =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/seq2edit/train.py \
#    --tune_bert 1 \
#    --lr 1e-5 \
#    --n_epoch 5 \
#    --batch_size ${BATCH_SIZE} \
#    --accumulation_size ${UPDATE_FREQ} \
#    --patience 3 \
#    --train_dataset "${DIR_INPUT_STAGE0}/train.char.label.shuf" \
#    --eval_dataset "${DIR_VALID}/valid.char.label" \
#    --output_dir ${DIR_MODEL_STAGE0} \
#    --model_name "checkpoint_best" \
#    --vocab_path ${VOCAB_PATH} \
#    --weights_name ${PRETRAIN_WEIGHTS_DIR} \
#    --model_name_or_path "models/seq2edit/exps/zho/freeze_struct_bert_large/E2_F_31.54.th" \
#    --eval_gec_dataset ${DATASET_VALID} \
#    --eval_start_epoch 1 \
#    --skip_correct 0 \
#    --seed ${SEED} \
#    >${DIR_MODEL_STAGE0}/nohup.log 2>&1 &
#wait

#    --model_name_or_path "${DIR_MODEL_STAGE0}/checkpoint_best_score.pt" \
# ========================= Seq2Edit stage 1 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/seq2edit/train.py \
    --tune_bert 1 \
    --lr 1e-5 \
    --n_epoch 5 \
    --batch_size ${BATCH_SIZE} \
    --accumulation_size ${UPDATE_FREQ} \
    --patience 5 \
    --train_dataset "${DIR_INPUT_STAGE1}/train.char.label.shuf" \
    --eval_dataset "${DIR_VALID}/valid.char.label" \
    --output_dir ${DIR_MODEL_STAGE1} \
    --model_name "checkpoint_best" \
    --vocab_path ${VOCAB_PATH} \
    --weights_name ${PRETRAIN_WEIGHTS_DIR} \
    --model_name_or_path "models/seq2edit/exps/zho/stage0/ccl_pn0.1*4+bt6.0*4/E1_F_54.76.pt" \
    --eval_gec_dataset ${DATASET_VALID} \
    --eval_start_epoch 1 \
    --skip_correct 0 \
    --seed ${SEED} \
    >${DIR_MODEL_STAGE1}/nohup.log 2>&1 &
wait

# ========================= Seq2Edit Stage 2 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/seq2edit/train.py \
    --tune_bert 1 \
    --lr 3e-6 \
    --n_epoch 3 \
    --batch_size ${BATCH_SIZE} \
    --accumulation_size ${UPDATE_FREQ} \
    --patience 5 \
    --train_dataset "models/seq2edit/preprocess/zho/yaclc_dev/train.char.label.shuf" \
    --eval_dataset "${DIR_VALID}/valid.char.label" \
    --output_dir ${DIR_MODEL_STAGE2} \
    --model_name "checkpoint_best" \
    --vocab_path ${VOCAB_PATH} \
    --weights_name ${PRETRAIN_WEIGHTS_DIR} \
    --model_name_or_path "${DIR_MODEL_STAGE1}/checkpoint_best_score.pt" \
    --eval_gec_dataset ${DATASET_VALID} \
    --eval_start_epoch 1 \
    --skip_correct 0 \
    --seed ${SEED} \
    >${DIR_MODEL_STAGE2}/nohup.log 2>&1 &
wait

bash scripts/seq2edit/predict_batch.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n "E1*" \
    -v yaclc_minimal_test

bash scripts/seq2edit/predict_batch.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n "E1*" \
    -v yaclc_fluency_test

bash scripts/seq2edit/predict_batch.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n "E1*" \
    -v nlpcc_test

bash scripts/seq2edit/predict_batch.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n "E1*" \
    -v mucgec_dev