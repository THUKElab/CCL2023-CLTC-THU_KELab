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
    r)
        RATE_MASKING=${OPTARG};;
    d)
        DATASET=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=${SEED:-"42"}

ARCH=${ARCH:-"gec_bart_large"}

MAX_TOKENS=1536

UPDATE_FREQ=12

RATE_MASKING=${RATE_MASKING:-"0.05"}

DIR_MODEL=${DIR_MODEL:-"news8M_bt+ccl_open_filter-cutoff0.05_reg1.0-src_drop0.2"}

DIR_VALID=${DIR_VALID:-"models/fairseq/bart/preprocess/zho/yaclc_minimal_dev"}

DATASET_VALID="yaclc_minimal_dev"

#DIR_MODEL_STAGE0=models/fairseq/bart/exps/zho/cutoff/stage0/${DIR_MODEL}
DIR_MODEL_STAGE1=models/fairseq/bart/exps/zho/cutoff/stage1/${DIR_MODEL}
DIR_MODEL_STAGE2=models/fairseq/bart/exps/zho/cutoff/stage2/${DIR_MODEL}

DIR_INPUT_STAGE1=models/fairseq/bart/preprocess/zho/real/ccl_open_filter_stage1
#DIR_INPUT_STAGE1=models/fairseq/bart/preprocess/zho/real/ccl_open_filter_stage1_denoise/mediate_ensemble_s5e5_6
DIR_INPUT_STAGE2=models/fairseq/bart/preprocess/zho/real/yaclc_dev

#mkdir -p ${DIR_MODEL_STAGE0}
mkdir -p ${DIR_MODEL_STAGE1}
mkdir -p ${DIR_MODEL_STAGE2}
#mkdir -p ${DIR_MODEL_STAGE0}/results
mkdir -p ${DIR_MODEL_STAGE1}/results
mkdir -p ${DIR_MODEL_STAGE2}/results


#if [ ! -d ${DIR_INPUT_STAGE0}/bin ]; then
#    bash scripts/fairseq/zho/preprocess.sh  -i ${DIR_INPUT_STAGE0}  -o ${DIR_INPUT_STAGE0}  -v ${DIR_VALID}
#fi

if [ ! -d ${DIR_INPUT_STAGE1}/bin ]; then
    bash scripts/fairseq/zho/preprocess.sh  -i ${DIR_INPUT_STAGE1}  -o ${DIR_INPUT_STAGE1}  -v ${DIR_VALID}
fi

if [ ! -d ${DIR_INPUT_STAGE2}/bin ]; then
    bash scripts/fairseq/zho/preprocess.sh  -i ${DIR_INPUT_STAGE2}  -o ${DIR_INPUT_STAGE2}  -v ${DIR_VALID}
fi


# TODO: 使用伪数据集训练
#  --restore-file transformers:fnlp/bart-large-chinese \
# ========================= CutOff stage 0 =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT_STAGE0}/bin \
#    --save-dir ${DIR_MODEL_STAGE0} \
#    --user-dir models/fairseq/bart \
#    --finetune-from-model models/fairseq/bart/exps/zho/pattern_noise/stage0/news8M_pattern_noise-src_drop0.2/checkpoint_best_score.pt \
#    --task gec_dev \
#    --arch ${ARCH} \
#    --max-tokens 1536 \
#    --max-source-positions 1024 \
#    --max-target-positions 1024 \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --update-freq ${UPDATE_FREQ} \
#    --lr 3e-05 \
#    --warmup-updates 2000 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.2 \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 1.0 \
#    --criterion augmented_label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 5 \
#    --patience 2 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --remove-bpe \
#    --eval-gec \
#    --eval-gec-min-update 2000 \
#    --eval-gec-metric "errant_zho" \
#    --eval-gec-dataset ${DATASET_VALID} \
#    --eval-gec-output-prefix ${DIR_MODEL_STAGE0}/results/output \
#    --beam 12 \
#    --augmentation-schema "cut_off" \
#    --augmentation-masking-probability ${RATE_MASKING} \
#    --augmentation-masking-schema "word" \
#    --regularization-weight 1.0 \
#    --num-workers 2 \
#    --seed ${SEED} >${DIR_MODEL_STAGE0}/nohup.log 2>&1 &
#wait


#    --finetune-from-model ${DIR_MODEL_STAGE0}/checkpoint_best.pt \
#    --finetune-from-model models/fairseq/bart/exps/zho/pattern_noise/stage0/news8M_pattern_noise-src_drop0.2/checkpoint_best_score.pt \
#    --restore-file transformers:fnlp/bart-large-chinese \
# ========================= CutOff stage 1 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT_STAGE1}/bin \
    --save-dir ${DIR_MODEL_STAGE1} \
    --user-dir models/fairseq/bart \
    --task gec_dev \
    --arch ${ARCH} \
    --restore-file transformers:fnlp/bart-large-chinese \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-tokens ${MAX_TOKENS} \
    --update-freq ${UPDATE_FREQ} \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 30 \
    --patience 5 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --augmentation-schema "cut_off" \
    --augmentation-masking-probability ${RATE_MASKING} \
    --augmentation-masking-schema "word" \
    --regularization-weight 1.0 \
    --num-workers 2 \
    --seed ${SEED} >${DIR_MODEL_STAGE1}/nohup.log 2>&1 &
wait

# ========================= CutOff Stage 2 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT_STAGE2}/bin \
    --save-dir ${DIR_MODEL_STAGE2} \
    --user-dir models/fairseq/bart \
    --finetune-from-model ${DIR_MODEL_STAGE1}/checkpoint_best_score.pt \
    --task gec_dev \
    --arch ${ARCH} \
    --max-tokens ${MAX_TOKENS} \
    --update-freq ${UPDATE_FREQ} \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --lr 2e-05 \
    --warmup-updates 100 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 3 \
    --patience 5 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --remove-bpe \
    --beam 12 \
    --augmentation-schema "cut_off" \
    --augmentation-masking-probability ${RATE_MASKING} \
    --augmentation-masking-schema "word" \
    --regularization-weight 1.0 \
    --num-workers 2 \
    --seed ${SEED} >${DIR_MODEL_STAGE2}/nohup.log 2>&1 &
wait

CHECKPOINT="checkpoint3.pt"

if [ ! -f ${DIR_MODEL_STAGE2}/${CHECKPOINT} ]; then
    echo "No specified checkpoint: ${DIR_MODEL_STAGE2}/${CHECKPOINT}"
    exit 1
fi

bash scripts/fairseq/zho/predict.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n ${CHECKPOINT} \
    -v yaclc_minimal_test

bash scripts/fairseq/zho/predict.sh -g ${GPU_list}\
    -m ${DIR_MODEL_STAGE2} \
    -n ${CHECKPOINT} \
    -v yaclc_fluency_test
