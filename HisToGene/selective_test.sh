#!/bin/bash

# Selective batch submission for HisToGene pathway prediction tests
# Only tests models that are already trained

PARTITION="gpuA"
MEMORY="64G"
TIME="12:00:00"
OUTPUT_DIR="./HisToGene_test_results"

mkdir -p "$OUTPUT_DIR"

START_FOLD=${1:-0}
END_FOLD=${2:-11}
SELECTED_SCORE=${3:-""}
PATHWAY_FILTER=${4:-""}   # GO/KEGG/MSigDB or empty for all
DATASET=${5:-"HER2"}

declare -A PATHWAYS
PATHWAYS[GO_0034340]="GO"
PATHWAYS[GO_0060337]="GO"
PATHWAYS[GO_0071357]="GO"
PATHWAYS[KEGG_P53_SIGNALING_PATHWAY]="KEGG"
PATHWAYS[KEGG_BREAST_CANCER]="KEGG"
PATHWAYS[KEGG_APOPTOSIS]="KEGG"
PATHWAYS[BREAST_CANCER_LUMINAL_A]="MSigDB"
PATHWAYS[HER2_AMPLICON_SIGNATURE]="MSigDB"
PATHWAYS[HALLMARK_MYC_TARGETS_V1]="MSigDB"

if [[ -n "$SELECTED_SCORE" ]]; then
    SCORES=("$SELECTED_SCORE")
else
    SCORES=("AUCell" "UCell")
fi

get_pathway_short() {
    case $1 in
        "GO_0034340") echo "G34" ;;
        "GO_0060337") echo "G337" ;;
        "GO_0071357") echo "G357" ;;
        "KEGG_P53_SIGNALING_PATHWAY") echo "KP53" ;;
        "KEGG_BREAST_CANCER") echo "KBC" ;;
        "KEGG_APOPTOSIS") echo "KA" ;;
        "BREAST_CANCER_LUMINAL_A") echo "BCLA" ;;
        "HER2_AMPLICON_SIGNATURE") echo "HER2" ;;
        "HALLMARK_MYC_TARGETS_V1") echo "MYC" ;;
        *) echo "UNK" ;;
    esac
}

check_model_exists() {
    local fold=$1
    local pathway=$2
    local score=$3
    local model_paths=(
        "./model/pathway/${score}/${pathway}/last_train_-htg_${pathway}_${score}_cv_${fold}.ckpt"
        "./model/last_train_-htg_${pathway}_${score}_cv_${fold}.ckpt"
        "./model_pathway_HisToGene/${score}/${pathway}/fold_$(printf "%02d" $fold)_${pathway}_${score}.ckpt"
    )
    for path in "${model_paths[@]}"; do
        [[ -f "$path" ]] && { echo "$path"; return 0; }
    done
    return 1
}

SUBMITTED=0
SKIPPED_NO_MODEL=0
SKIPPED_FILTER=0
FOUND_MODELS=0

for fold in $(seq $START_FOLD $END_FOLD); do
    for pathway in "${!PATHWAYS[@]}"; do
        pathway_type="${PATHWAYS[$pathway]}"
        if [[ -n "$PATHWAY_FILTER" && "$pathway_type" != "$PATHWAY_FILTER" ]]; then
            ((SKIPPED_FILTER++))
            continue
        fi
        for score in "${SCORES[@]}"; do
            if MODEL_PATH=$(check_model_exists $fold $pathway $score); then
                ((FOUND_MODELS++))
                RESULT_DIR="./results_HisToGene_pathway/${DATASET}/${score}/${pathway}"
                RESULT_FILE="${RESULT_DIR}/fold_$(printf "%02d" $fold)_${pathway}_${score}_results.txt"
                if [[ -f "$RESULT_FILE" ]]; then
                    ((SKIPPED_NO_MODEL++))
                    continue
                fi
                pathway_short=$(get_pathway_short "$pathway")
                score_short="${score:0:1}"
                JOB_NAME="TEST_HTG_f${fold}_${pathway_short}_${score_short}"
                OUTPUT_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" $fold)_${pathway_short}_${score_short}_%j.out"
                ERROR_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" $fold)_${pathway_short}_${score_short}_%j.err"
                TEST_CMD="python -u test_1_0818.py --fold $fold --pathway $pathway --score $score --dataset $DATASET --device auto --save_results"
                sbatch -p $PARTITION \
                       --gres=gpu:1 \
                       --mem=$MEMORY \
                       --time=$TIME \
                       --job-name=$JOB_NAME \
                       --output=$OUTPUT_FILE \
                       --error=$ERROR_FILE \
                       --wrap="$TEST_CMD"
                ((SUBMITTED++))
                echo "Submitted: fold $fold | $pathway ($pathway_type) | $score"
                sleep 0.1
            else
                ((SKIPPED_NO_MODEL++))
            fi
        done
    done
done

echo "Jobs submitted: $SUBMITTED | Models found: $FOUND_MODELS | Skipped(no model/results): $SKIPPED_NO_MODEL | Skipped(filter): $SKIPPED_FILTER"
echo "Logs: $OUTPUT_DIR/"
echo "Results: ./results_HisToGene_pathway/${DATASET}/{score}/{pathway}/"
