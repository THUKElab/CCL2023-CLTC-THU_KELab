#!/bin/bash

while getopts "g:m:n:i:o:v:b:r:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
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
    b)
        BIAS_KEEP=${OPTARG};;
    r)
        ROUND=${OPTARG};;
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

SEED=42

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_MODEL}/results"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best_score.pt"}

PATH_MODEL=${DIR_MODEL}/${FILE_MODEL}

FILE_LOG="${DIR_OUTPUT}/${VALID_NAME}.log"

BIAS_KEEP=${BIAS_KEEP:-"0.10"}

ROUND=${ROUND:-"5"}

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

# Default file for validation
if [ ${VALID_NAME} = "mucgec_dev" ]; then
    # Vocab Version 2
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/mucgec/dev/MuCGEC_dev.src"}
elif [ ${VALID_NAME} = "mucgec_test" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/mucgec/test/MuCGEC_test.src"}
elif [ ${VALID_NAME} = "nlpcc_test" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/NLPCC/test/source.txt"}
elif [ ${VALID_NAME} = "yaclc_minimal_dev" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/YACLC/dev/yaclc-minimal_dev.src"}
elif [ ${VALID_NAME} = "yaclc_fluency_dev" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/YACLC/dev/yaclc-fluency_dev.src"}
elif [ ${VALID_NAME} = "yaclc_minimal_test" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/YACLC/test/yaclc-minimal_test.src"}
elif [ ${VALID_NAME} = "yaclc_fluency_test" ]; then
    FILE_INPUT=${FILE_INPUT:-"../datasets/GEC/CGEC/YACLC/test/yaclc-fluency_test.src"}
else
    echo "Unknown VALID_NAME: ${VALID_NAME}"
fi

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model:  ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}
echo "Valid:  ${VALID_NAME}" | tee -a ${FILE_LOG}
echo "Bias:   ${BIAS_KEEP}" | tee -a ${FILE_LOG}
echo "Round:  ${ROUND}" | tee -a ${FILE_LOG}


CUDA_VISIBLE_DEVICES=${GPU_list} python models/seq2edit/predict.py \
    --model_path ${PATH_MODEL} \
    --input_file ${FILE_INPUT} \
    --output_file ${DIR_OUTPUT}/${VALID_NAME}.out \
    --iteration_count ${ROUND} \
    --keep_bias ${BIAS_KEEP} \
    --seed ${SEED}


# Evaluation
if [ ${VALID_NAME} = "mucgec_dev" ]; then
    echo "Post-process MuCGEC_dev" | tee -a ${FILE_LOG}
    FILE_REF="../datasets/GEC/CGEC/mucgec/dev/MuCGEC_dev.m2"

#    python utils/post_processors/post_process_chinese.py ${FILE_INPUT} \
#        ${DIR_OUTPUT}/${VALID_NAME}.out \
#        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
#        ${FILE_ID} 10000

    # ERRANT Evaluation
    echo "ERRANT Evaluation" | tee -a ${FILE_LOG}
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_INPUT} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "mucgec_test" ]; then
    echo "Post-process MuCGEC_test" | tee -a ${FILE_LOG}

    python utils/post_processors/post_process_chinese.py ${FILE_INPUT} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 100000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed.para

elif [ ${VALID_NAME} = "nlpcc_test" ]; then
    echo "Post-process nlpcc_test" | tee -a ${FILE_LOG}
    FILE_REF="../datasets/GEC/CGEC/NLPCC/test/gold/gold.01"

#    python utils/post_processors/post_process_chinese.py ${FILE_INPUT} \
#        ${DIR_OUTPUT}/${VALID_NAME}.out \
#        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
#        ${FILE_ID} 128

    # Word Segment
    python ../libgrass-ui/main.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.word \

    # MaxMatch Evaluation
    echo "M2 Evaluation" | tee -a ${FILE_LOG}
    python metrics/m2/m2scorer.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out.word \
        ${FILE_REF} | tee -a ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_minimal_dev" ]; then
    echo "Post-process yaclc_minimal_dev" | tee -a ${FILE_LOG}
    FILE_REF="../datasets/GEC/CGEC/YACLC/dev/yaclc-minimal_dev.m2"

#    python utils/post_processors/post_process_chinese.py ${FILE_INPUT} \
#        ${DIR_OUTPUT}/${VALID_NAME}.out \
#        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
#        ${FILE_ID} 10000

    # ERRANT Evaluation
    echo "ERRANT Evaluation" | tee -a ${FILE_LOG}
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_INPUT} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_fluency_dev" ]; then
    echo "Post-process yaclc_fluency_dev" | tee -a ${FILE_LOG}
    FILE_REF="../datasets/GEC/CGEC/YACLC/dev/yaclc-fluency_dev.m2"

#    python utils/post_processors/post_process_chinese.py ${FILE_INPUT} \
#        ${DIR_OUTPUT}/${VALID_NAME}.out \
#        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
#        ${FILE_ID} 10000

    # ERRANT Evaluation
    echo "ERRANT Evaluation" | tee -a ${FILE_LOG}
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_INPUT} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_minimal_test" ]; then
    echo "Post-process yaclc_minimal_test" | tee -a ${FILE_LOG}
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_minimal_test/yaclc-minimal_test.id"
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_minimal_test/yaclc-minimal_test.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post1 \
        ${FILE_ID} 10000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post1 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-minimal_test.para1

    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-minimal_test.para1 \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        -g char | tee -a ${FILE_LOG}

    python utils/post_processors/post_process_yaclc.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post2

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post2 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-minimal_test.para

    # 用于集成
    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-minimal_test.para \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant2 \
        -g char | tee -a ${FILE_LOG}

elif [ ${VALID_NAME} = "yaclc_fluency_test" ]; then
    echo "Post-process yaclc_fluency_test" | tee -a ${FILE_LOG}
    FILE_ID="models/fairseq/bart/preprocess/zho/yaclc_fluency_test/yaclc-fluency_test.id"
    FILE_SRC="models/fairseq/bart/preprocess/zho/yaclc_fluency_test/yaclc-fluency_test.src"

    python utils/post_processors/post_process_chinese.py ${FILE_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post1 \
        ${FILE_ID} 10000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post1 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-fluency_test.para1

    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-fluency_test.para1 \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        -g char | tee -a ${FILE_LOG}

    python utils/post_processors/post_process_yaclc.py \
        ${DIR_OUTPUT}/${VALID_NAME}.out.errant \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post2

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post2 \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/yaclc-fluency_test.para

    # 用于集成
    python metrics/ChERRANT/parallel_to_m2.py \
        -f ${DIR_OUTPUT}/yaclc-fluency_test.para \
        -o ${DIR_OUTPUT}/${VALID_NAME}.out.errant2 \
        -g char | tee -a ${FILE_LOG}

fi


# ============================== MuCGEC_dev Results ============================== #
#                                     Segment Keep  Round Simply  post   P       R       F0.5    eval_loss   TP    FP    FN
# MuCGEC_Official                         N   0.00  4     N              39.76   26.40   36.10               1081  1638  3014
# MuCGEC_Official                         N   0.00  4     Y              39.79   26.45   36.14               1083  1639  3012
# MuCGEC_Official                         N   0.05  4     Y              40.73   25.72   36.48               1048  1525  3026
# MuCGEC_Official                         N   0.10  4     Y              41.59   24.80   36.63               1004  1410  3045
# MuCGEC_Official                         N   0.10  5     N              41.60   24.77   36.62               1008  1415  3062
# MuCGEC_Official                         N   0.10  5     Y              41.66   24.79   36.67               1009  1413  3061
# MuCGEC_Official                         N   0.10  6     Y              41.61   24.86   36.67               1007  1413  3043
# MuCGEC_Official                         N   0.15  4     Y              41.50   23.53   36.00               945   1332  3072

# ============================== YACLC_minimal_dev Results ============================== #
# MuCGEC_Official                         N  0.00  4     Y              61.19   41.44   55.86               1769  1122  2500
# MuCGEC_Official                         N  0.00  5     Y              61.47   41.54   56.09               1774  1112  2497
# MuCGEC_Official                         Y  0.00  5     Y              61.67   41.66   56.27               1786  1110  2501
# MuCGEC_Official                         N  0.05  4     Y              62.46   38.65   55.61               1587  954   2519
# MuCGEC_Official                         N  0.05  5     Y              62.69   38.61   55.73               1591  947   1591

# CCL-LR1e-5                              N  0.05  4     Y              62.70   35.41   54.32               1395  830   2545
# CCL-LR2e-5                              N  0.05  4     Y              65.39   32.29   54.26               1194  632   2504

# CCL_ERR-LR3e-6                          N  0.05  4     Y              60.65   37.63   54.04               1535  996   2544
# CCL_ERR-LR5e-6                          N  0.05  4     Y              61.54   36.65   54.18               1491  932   2577
# CCL_ERR-LR8e-6                          N  0.05  4     Y              57.92   38.21   52.50               1595  1159  2579
# CCL_ERR-LR1e-5                          N  0.05  4     Y              61.92   35.35   53.83               1384  851   2531
# CCL_ERR-LR2e-5                          N  0.05  4     Y              62.65   36.09   54.61               1419  846   2513


# ============================== YACLC test Restricted Results ============================== #
#                                     Segment Keep  Round Simply  post    P       R       F0.5      P       R       F0.5    Average
# CCL_pn0.1*4+bt6.0*4-stage1_ckpt01       Y   0.05  4     Y       Y       68.51   39.09   59.54     45.91   16.69   34.01   46.78
# CCL_denoise-stage2_ckpt01               Y   0.05  4     Y       Y       71.51   48.99   65.49     46.59   22.47   38.36   51.83
# CCL_pn_bt+CCL_denoise-stage2_ckpt01_0   Y   0.05  4     Y       Y       72.27   52.19   67.11     47.61   24.26   39.93   53.52
# CCL_pn_bt+CCL_denoise-stage2_ckpt01_1   Y   0.05  4     Y       Y       72.10   52.76   67.17     47.43   25.01   40.22   53.70
# CCL_pn_bt+CCL_denoise-stage2_ckpt01_2   Y   0.05  4     Y       Y       71.97   52.41   66.97     47.30   24.43   39.84   53.41
# CCL_pn_bt+CCL_denoise-stage2_ckpt01_42  Y   0.05  4     Y       Y       72.00   51.89   66.82     47.39   24.11   39.72   53.27

# ccl_pn_bt+ccl_denoise_ensemble_s5e5_6   Y   0.05  4     Y       Y       72.89   52.35   67.58     47.98   24.72   40.38   53.98


# ============================== YACLC test Open Results ============================== #
# news8M_pn+ccl_pn0.1*4+bt6.0*4_ckpt01    Y   0.05  4     Y               69.61   52.14   65.24     45.16   23.80   38.29   51.77
# news8M_pn+ccl_pn0.1*4+bt6.0*4_ckpt01    Y   0.05  4     Y       Y       69.16   62.13   67.63     41.70   28.95   38.32   52.97
# news8M_pn+ccl_pn0.1*4+bt6.0*4_ckpt03    Y   0.05  4     Y       Y       69.87   54.31   66.08     45.27   25.55   39.22   52.65
# news8M_pn+ccl_pn0.1*4+bt6.0*4_ckpt05    Y   0.05  4     Y       Y       69.29   54.71   65.78     44.74   25.93   39.07   52.43

# ccl_pn_bt+ccl_open_filter_denoise-seed0 Y   0.05  4     Y       Y       74.11   52.16   68.36     49.48   23.73   40.65   54.50
# ccl_pn_bt+ccl_open_filter_denoise-seed1 Y   0.05  4     Y       Y       73.92   52.15   68.23     48.86   23.68   40.29   54.26
# ccl_pn_bt+ccl_open_filter_denoise-seed2 Y   0.05  4     Y       Y       74.25   51.88   68.36     49.20   23.45   40.34   54.35
# ccl_pn_bt+ccl_open_filter_denoise-seed3 Y   0.05  4     Y       Y
# ccl_pn_bt+ccl_open_filter_denoise-seed4 Y   0.05  4     Y       Y

# ccl_pn_bt+ccl_open_filter_denoise_ensemble_s5e5_6-seed0
#                                         Y   0.00  4     Y       Y       72.83   53.78   68.01     48.08   25.06   40.61   54.31
#                                         Y   0.05  4     Y       Y       74.20   52.04   68.38     49.33   24.00   40.73   54.55
#                                         Y   0.10  5     Y       Y       75.56   50.11   68.59     50.41   22.67   40.50   54.55
#                                         Y   0.15  5     Y       Y
#                                         Y   0.20  6     Y       Y       70.65   55.37   66.95     53.01   20.40   40.17   53.56

# ccl_pn_bt+ccl_open_filter_denoise_ensemble_s5e5_6_stage1-seed1
#                                         Y   0.05  4     Y       Y       71.62   53.52   67.08     46.42   25.00   39.63   53.36
# ccl_pn_bt+ccl_open_filter_denoise_ensemble_s5e5_6_stage2-seed1
#                                         Y   0.05  4     Y       Y       74.31   51.99   68.43     48.99   23.74   40.40   54.42
