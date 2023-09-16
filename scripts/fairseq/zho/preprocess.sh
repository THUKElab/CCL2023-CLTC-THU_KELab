#!/bin/bash

while getopts "p:t:i:o:v:" optname; do
    case $optname in
    p)
        DIR_PREPROCESS=${OPTARG};;
    t)
        FILE_TSV=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    v)
        DIR_VALID=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

if [ -z ${DIR_INPUT} ]; then
    echo "DIR_INPUT not exists"
    exit -1
fi

DIR_PREPROCESS=${DIR_PREPROCESS:-"models/fairseq/bart/preprocess/zho"}

DIR_VALID=${DIR_VALID:-""}

DIR_OUTPUT=${DIR_OUTPUT:-${DIR_INPUT}}

# Optional: Split TSV file to SRC and TGT files
if [ -f "${FILE_TSV}" ] && [ ! -f "${DIR_INPUT}/src.txt" ]; then
    awk -F'\t' '{print $1}' ${FILE_TSV} > "${DIR_INPUT}/src.txt"
    awk -F'\t' '{print $2}' ${FILE_TSV} > "${DIR_INPUT}/tgt.txt"
fi

# Preprocess training dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_OUTPUT}/train.char.${LANG}" ]; then
        echo "Segment: ${DIR_INPUT}/${LANG}.txt -> ${DIR_OUTPUT}/train.char.${LANG}"
        python models/fairseq/bart/preprocess/zho/segment_bert.py \
            < "${DIR_INPUT}/${LANG}.txt" \
            > "${DIR_OUTPUT}/train.char.${LANG}"
    fi
done

# Preprocess validation dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_VALID}/valid.char.${LANG}" ]; then
        echo "Segment: ${DIR_VALID}/${LANG}.txt -> ${DIR_VALID}/valid.char.${LANG}"
        python models/fairseq/bart/preprocess/zho/segment_bert.py \
            < "${DIR_VALID}/${LANG}.txt" \
            > "${DIR_VALID}/valid.char.${LANG}"
    fi
done

if [ ! -d "${DIR_OUTPUT}/bin" ]; then
    fairseq-preprocess --source-lang "src" --target-lang "tgt" \
        --trainpref "${DIR_OUTPUT}/train.char" \
        --validpref "${DIR_VALID}/valid.char" \
        --destdir "${DIR_OUTPUT}/bin/" \
        --workers 64 \
        --srcdict ${DIR_PREPROCESS}/vocab_count_v2.txt \
        --tgtdict ${DIR_PREPROCESS}/vocab_count_v2.txt
fi
