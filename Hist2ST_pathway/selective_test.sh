#!/bin/bash

# Selective batch submission for Hist2ST tests (only for trained models)

# Usage:
# bash selective_test.sh                      # all trained models
# bash selective_test.sh 0 5                  # folds 0–5
# bash selective_test.sh 0 5 AUCell           # folds 0–5, AUCell
# bash selective_test.sh 0 5 AUCell GO        # folds 0–5, AUCell, GO pathways

PARTITION="gpuA"
MEMORY="64G"
TIME="12:00:00"
OUTPUT_DIR="./Hist2ST_test_results"

mkdir -p "$OUTPUT_DIR"

# Args
START_FOLD=${1:-0}
END_FOLD=${2:-35}
SELECTED_SCORE=${3:-""}
PATHWAY_FILTER=${4:-""}  # GO/KEGG/MSigDB or empty for all

# Pathways
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

# Scores
if [[ -n "$SELECTED_SCORE" ]]; then
    SCORES=("$SELECTED_SCORE")
else
    SCORES=("AUCell" "UCell")
fi

echo "Submitting Hist2ST test jobs..."
echo "Folds: $START_FOLD-$END_FOLD | Scores: ${SCORES[*]} | Pathway filter: ${PATHWAY_FILTER:-All}"

# Short names
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
            MODEL_PATH="./model/${score}/${pathway}/fold_$(printf "%02d" "$fold")_${pathway}_${score}.ckpt"

            if [[ ! -f "$MODEL_PATH" ]]; then
                ((SKIPPED++))
                echo "Skipped: fold $fold | $pathway ($pathway_type) | $score (model missing)"
                continue
            fi

            pathway_short=$(get_pathway_short "$pathway")
            score_short="${score:0:1}"

            JOB_NAME="TEST_f${fold}_${pathway_short}_${score_short}"
            OUTPUT_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" "$fold")_${pathway_short}_${score_short}_%j.out"
            ERROR_FILE="${OUTPUT_DIR}/test_fold_$(printf "%02d" "$fold")_${pathway_short}_${score_short}_%j.err"

            sbatch -p "$PARTITION" \
                   --gres=gpu:1 \
                   --mem="$MEMORY" \
                   --time="$TIME" \
                   --job-name="$JOB_NAME" \
                   --output="$OUTPUT_FILE" \
                   --error="$ERROR_FILE" \
                   --wrap="python -u test_1_0816.py --fold $fold --pathway $pathway --score $score"

            ((SUBMITTED++))
            echo "Submitted: fold $fold | $pathway ($pathway_type) | $score"
            sleep 0.1
        done
    done
done

echo "Done. Submitted: $SUBMITTED | Skipped: $SKIPPED"
