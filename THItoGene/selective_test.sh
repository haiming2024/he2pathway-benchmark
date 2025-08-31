#!/bin/bash

# Selective batch submission for THItoGene testing
# Only tests models that have already been trained

PARTITION="gpuA"
MEMORY="64G"
TIME="12:00:00"
OUTPUT_DIR="./THItoGene_test_results"

mkdir -p "$OUTPUT_DIR"

START_FOLD=${1:-0}
END_FOLD=${2:-35}
SELECTED_SCORE=${3:-""}
PATHWAY_FILTER=${4:-""}  # GO/KEGG/MSigDB or empty for all

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

SUBMITTED=0
SKIPPED=0

for fold in $(seq "$START_FOLD" "$END_FOLD"); do
    for pathway in "${!PATHWAYS[@]}"; do
        pathway_type="${PATHWAYS[$pathway]}"
        if [[ -n "$PATHWAY_FILTER" && "$pathway_type" != "$PATHWAY_FILTER" ]]; then
            continue
        fi
        for score in "${SCORES[@]}"; do
            MODEL_PATH="./model_pathway_THItoGene/${score}/${pathway}/fold_$(printf "%02d" "$fold")_${pathway}_${score}.ckpt"
            if [[ ! -f "$MODEL_PATH" ]]; then
                ((SKIPPED++))
                continue
            fi
            pathway_short=$(get_pathway_short "$pathway")
            score_short="${score:0:1}"
            JOB_NAME="TEST_THI_f${fold}_${pathway_short}_${score_short}"
            OUTPUT_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" "$fold")_${pathway_short}_${score_short}_%j.out"
            ERROR_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" "$fold")_${pathway_short}_${score_short}_%j.err"

            sbatch -p "$PARTITION" \
                   --gres=gpu:1 \
                   --mem="$MEMORY" \
                   --time="$TIME" \
                   --job-name="$JOB_NAME" \
                   --output="$OUTPUT_FILE" \
                   --error="$ERROR_FILE" \
                   --wrap="python -u test_1_0817.py --fold $fold --pathway $pathway --score $score"

            ((SUBMITTED++))
            echo "Submitted fold $fold | $pathway | $score"
            sleep 0.1
        done
    done
done

echo "Submitted: $SUBMITTED"
echo "Skipped (no model): $SKIPPED"
