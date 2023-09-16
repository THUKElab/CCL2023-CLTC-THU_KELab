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

DIR_PREPROCESS=${DIR_PREPROCESS:-"models/seq2edit/preprocess/zho"}

DIR_OUTPUT=${DIR_OUTPUT:-${DIR_INPUT}}

DIR_VALID=${DIR_VALID:-"models/seq2edit/preprocess/zho/mucgec_dev"}

# Optional: Split TSV file to SRC and TGT files
if [ -f "${FILE_TSV}" ] && [ ! -f "${DIR_INPUT}/src.txt" ]; then
    awk -F'\t' '{print $1}' ${FILE_TSV} > "${DIR_INPUT}/src.txt"
    awk -F'\t' '{print $2}' ${FILE_TSV} > "${DIR_INPUT}/tgt.txt"
fi

# Preprocess training dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_OUTPUT}/train.char.${LANG}" ]; then
        echo "Segment: ${DIR_INPUT}/${LANG}.txt -> ${DIR_OUTPUT}/train.char.${LANG}"
        python models/seq2edit/preprocess/zho/segment_bert.py \
            < "${DIR_INPUT}/${LANG}.txt" \
            > "${DIR_OUTPUT}/train.char.${LANG}"
    fi
done

# Preprocess validation dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_VALID}/valid.char.${LANG}" ]; then
        echo "Segment: ${DIR_VALID}/valid.${LANG} -> ${DIR_VALID}/valid.char.${LANG}"
        python models/seq2edit/preprocess/zho/segment_bert.py \
            < "${DIR_VALID}/valid.${LANG}" \
            > "${DIR_VALID}/valid.char.${LANG}"
    fi
done

# Generate training label file
if [ ! -f ${DIR_OUTPUT}/train.char.label.shuf ]; then
    python models/seq2edit/preprocess/zho/preprocess_data.py \
        -s "${DIR_OUTPUT}/train.char.src" \
        -t "${DIR_OUTPUT}/train.char.tgt" \
        -o "${DIR_OUTPUT}/train.char.label" \
        --worker_num 128

    shuf ${DIR_OUTPUT}/train.char.label > ${DIR_OUTPUT}/train.char.label.shuf
fi

# Generate validation label file
if [ ! -f ${DIR_VALID}/valid.char.label ]; then
    python models/seq2edit/preprocess/zho/preprocess_data.py \
        -s "${DIR_VALID}/valid.char.src" \
        -t "${DIR_VALID}/valid.char.tgt" \
        -o "${DIR_VALID}/valid.char.label" \
        --worker_num 128
fi
