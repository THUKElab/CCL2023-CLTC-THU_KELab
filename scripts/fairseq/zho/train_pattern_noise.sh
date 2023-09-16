#!/bin/bash

while getopts "g:i:o:v:m:r:" optname; do
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
    r)
        RATE_MASKING=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

ARCH=${ARCH:-"gec_bart_large"}

DIR_MODEL=${DIR_MODEL:-"bart_src_drop0.2"}

DIR_MODEL_STAGE0=models/fairseq/bart/exps/zho/pattern_noise/stage0/${DIR_MODEL}
DIR_MODEL_STAGE1=models/fairseq/bart/exps/zho/pattern_noise/stage1/${DIR_MODEL}

DIR_INPUT_STAGE0=models/fairseq/bart/preprocess/zho/pattern_noise/news8M
DIR_INPUT_STAGE1=models/fairseq/bart/preprocess/zho/real/hsk+lang8

mkdir -p ${DIR_MODEL_STAGE0}
mkdir -p ${DIR_MODEL_STAGE1}
mkdir -p ${DIR_MODEL_STAGE0}/results
mkdir -p ${DIR_MODEL_STAGE1}/results

# ========================= Bart-setting stage 0 =========================
# 相比 SynGEC 的改动：
# --dropout 0.1 -> 0.2
# --update-freq 8 -> 4
# --max-tokens 2048 -> 4096
# --warmup-updates 2000
# --layernorm-embedding True -> True
# --max-source-positions 512 -> 1024
# --max-target-positions 512 -> 1024
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT_STAGE0}/bin \
    --save-dir ${DIR_MODEL_STAGE0} \
    --user-dir models/fairseq/bart \
    --restore-file transformers:fnlp/bart-large-chinese \
    --task gec_dev \
    --arch ${ARCH} \
    --max-tokens 4096 \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 4 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 10 \
    --patience 5 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --remove-bpe \
    --eval-gec \
    --eval-gec-min-update 1000 \
    --eval-gec-metric errant_zho \
    --eval-gec-dataset mucgec_dev \
    --eval-gec-output-prefix ${DIR_MODEL_STAGE0}/results/output \
    --beam 12 \
    --seed 42 >${DIR_MODEL_STAGE0}/nohup.log 2>&1 &
wait

# ========================= Bart-setting stage 1 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT_STAGE1}/bin \
    --save-dir ${DIR_MODEL_STAGE1} \
    --user-dir models/fairseq/bart \
    --finetune-from-model ${DIR_MODEL_STAGE0}/checkpoint_best_score.pt \
    --task gec_dev \
    --arch ${ARCH} \
    --max-tokens 4096 \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 4 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 10 \
    --patience 5 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
    --remove-bpe \
    --eval-gec \
    --eval-gec-min-update 1000 \
    --eval-gec-metric errant_zho \
    --eval-gec-dataset mucgec_dev \
    --eval-gec-output-prefix ${DIR_MODEL_STAGE1}/results/output \
    --beam 12 \
    --seed 42 >${DIR_MODEL_STAGE1}/nohup.log 2>&1 &
wait


bash scripts/fairseq/zho/predict.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE1} \
    -n checkpoint_best_score.pt \
    -v nlpcc_test

bash scripts/fairseq/zho/predict.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE1} \
    -n checkpoint_best_score.pt \
    -v mucgec_test

