#!/bin/bash

while getopts "s:h:m:d:r:l:" optname; do
    case $optname in
    s)
        FILE_SRC=${OPTARG};;
    h)
        FILE_HYP=${OPTARG};;
    m)
        FILE_M2=${OPTARG};;
    d)
        DIR_HYP=${OPTARG};;
    r)
        FILE_REF=${OPTARG};;
    l)
        FILE_LOG=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SUFFIX_M2="errant"

DIR_DATASET="../datasets/GEC/CGEC/mucgec/dev"

FILE_SRC=${FILE_SRC:-"${DIR_DATASET}/MuCGEC_dev.char.src"}
FILE_REF=${FILE_REF:-"${DIR_DATASET}/MuCGEC_dev.m2"}
FILE_LOG=${FILE_LOG:-"temp.txt"}


if [ -z ${DIR_HYP} ]; then
    if [ -z ${FILE_HYP} ]; then
        echo "No DIR_HYP or FILE_HYP specified. Exit."
    else
        FILE_PARA=${FILE_HYP%.*}.para
        FILE_M2=${FILE_M2:-${FILE_HYP%.*}.${SUFFIX_M2}}

        echo "#################### ERRANT_ZHO Evaluation ####################" | tee -a ${FILE_LOG}
        echo "Source: ${FILE_SRC}" | tee -a ${FILE_LOG}
        echo "Reference: ${FILE_REF}" | tee -a ${FILE_LOG}
        echo "Hypothesis: ${FILE_HYP} -> ${FILE_PARA} -> ${FILE_M2}" | tee -a ${FILE_LOG}

        # Step1: Extract edits from hypothesis file
        paste ${FILE_SRC} ${FILE_HYP} | awk '{print NR"\t"$p}' > ${FILE_PARA}
        python metrics/ChERRANT/parallel_to_m2.py -f ${FILE_PARA} -o ${FILE_M2} -g char | tee -a ${FILE_LOG}

        # Step2: Compare hypothesis edits with reference edits
        python metrics/ChERRANT/compare_m2_for_evaluation.py -hyp ${FILE_M2} -ref ${FILE_REF} | tee -a ${FILE_LOG}
    fi
else
    for FILE_HYP in ${DIR_HYP}/pred_*.tgt; do
        FILE_M2=${FILE_HYP%.*}.${SUFFIX_M2}

        echo "#################### ERRANT_ZHO Evaluation ####################" | tee -a ${FILE_LOG}
        echo "Source: ${FILE_HYP}" | tee -a ${FILE_LOG}
        echo "Reference: ${FILE_REF}" | tee -a ${FILE_LOG}
        echo "Hypothesis: ${FILE_HYP} -> ${FILE_M2}" | tee -a ${FILE_LOG}

        # Step1: Extract edits from hypothesis file
        paste ${FILE_SRC} ${FILE_HYP} | awk '{print NR"\t"$p}' > ${FILE_PARA}
        python metrics/ChERRANT/parallel_to_m2.py -f ${FILE_PARA} -o ${FILE_M2} -g char | tee -a ${FILE_LOG}

        # Step2: Compare hypothesis edits with reference edits
        python metrics/ChERRANT/compare_m2_for_evaluation.py -hyp ${FILE_M2} -ref ${FILE_REF} | tee -a ${FILE_LOG}
    done
fi

