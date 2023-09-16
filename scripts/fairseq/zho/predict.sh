#!/bin/bash

while getopts "g:p:m:n:i:o:r:v:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    i)
        FILE_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    v)
        VALID_NAME=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

if [ -z ${DIR_MODEL} ]; then
    echo "DIR_MODEL not exists"
    exit -1
fi

DIR_PROCESSED=${DIR_PROCESSED:-"models/fairseq/bart/preprocess/zho/real/yaclc_dev/bin"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_MODEL}/results"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best.pt"}

FILE_LOG="${DIR_OUTPUT}/${VALID_NAME}.log"

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

# Default file for validation
if [ ${VALID_NAME} = "yaclc_minimal_dev" ]; then
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_minimal_dev/yaclc-minimal_dev.id"
    FILE_INPUT=${FILE_INPUT:-"models/fairseq/bart/preprocess/zho/yaclc_minimal_dev/yaclc-minimal_dev.char.src"}
elif [ ${VALID_NAME} = "yaclc_fluency_dev" ]; then
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_fluency_dev/yaclc-fluency_dev.id"
    FILE_INPUT=${FILE_INPUT:-"models/fairseq/bart/preprocess/zho/yaclc_fluency_dev/yaclc-fluency_dev.char.src"}
elif [ ${VALID_NAME} = "yaclc_minimal_test" ]; then
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_minimal_test/yaclc-minimal_test.id"
    FILE_INPUT=${FILE_INPUT:-"models/fairseq/bart/preprocess/zho/yaclc_minimal_test/yaclc-minimal_test.char.src"}
elif [ ${VALID_NAME} = "yaclc_fluency_test" ]; then
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_fluency_test/yaclc-fluency_test.id"
    FILE_INPUT=${FILE_INPUT:-"models/fairseq/bart/preprocess/zho/yaclc_fluency_test/yaclc-fluency_test.char.src"}
fi


echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model: ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Valid: ${VALID_NAME}" | tee -a ${FILE_LOG}
echo "Input: ${FILE_INPUT}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}

# Generate Hypothesis
N_BEST=1
CUDA_VISIBLE_DEVICES=${GPU_list} fairseq-interactive ${DIR_PROCESSED} \
    --task translation \
    --user-dir models/fairseq/bart \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 12 \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --buffer-size 10000 \
    --batch-size 128 \
    --num-workers 8 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    < ${FILE_INPUT} > ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | tee -a ${FILE_LOG}

cat ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | grep "^D-"  \
    | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" \
    | cut -f 3 > ${DIR_OUTPUT}/${VALID_NAME}.out
sed -i '$d' ${DIR_OUTPUT}/${VALID_NAME}.out


# Evaluation
if [ ${VALID_NAME} = "yaclc_minimal_dev" ]; then
    echo "Post-process yaclc_minimal_dev" | tee -a ${FILE_LOG}
    FILE_REF="models/fairseq/bart/preprocess/zho/yaclc_minimal_dev/yaclc-minimal_dev.m2"
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_minimal_dev/yaclc-minimal_dev.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    # ERRANT Evaluation
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_SRC} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_fluency_dev" ]; then
    echo "Post-process yaclc_fluency_dev" | tee -a ${FILE_LOG}
    FILE_REF="models/fairseq/bart/preprocess/zho/yaclc_fluency_dev/yaclc-fluency_dev.m2"
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_fluency_dev/yaclc-fluency_dev.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    # ERRANT Evaluation
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_SRC} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_minimal_test" ]; then
    echo "Post-process yaclc_minimal_test" | tee -a ${FILE_LOG}
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_minimal_test/yaclc-minimal_test.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-minimal_test.para0

    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-minimal_test.para0 \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        -g char | tee -a ${FILE_LOG}

    python utils/post_processors/post_process_yaclc.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed2

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed2 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-minimal_test.para

    # 用于集成
    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-minimal_test.para \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant2 \
        -g char | tee -a ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_fluency_test" ]; then
    echo "Post-process yaclc_fluency_test" | tee -a ${FILE_LOG}
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_fluency_test/yaclc-fluency_test.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-fluency_test.para0

    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-fluency_test.para0 \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        -g char | tee -a ${FILE_LOG}

    python utils/post_processors/post_process_yaclc.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed2

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed2 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-fluency_test.para

    # 用于集成
    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-fluency_test.para \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant2 \
        -g char | tee -a ${FILE_LOG}
fi

