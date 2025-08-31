#!/bin/bash

# Summarize all THItoGene test results

RESULTS_DIR="./results_THItoGene"
SUMMARY_FILE="./THItoGene_pathway_performance_summary.csv"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Results directory does not exist: $RESULTS_DIR"
    exit 1
fi

echo "Model,ScoreMethod,Pathway,Fold,TestSample,PCC,PValue,NSpots" > $SUMMARY_FILE

TOTAL_FILES=0
PROCESSED=0

for score_dir in "$RESULTS_DIR"/*; do
    if [[ -d "$score_dir" ]]; then
        score_method=$(basename "$score_dir")
        for pathway_dir in "$score_dir"/*; do
            if [[ -d "$pathway_dir" ]]; then
                pathway=$(basename "$pathway_dir")
                for result_file in "$pathway_dir"/*.txt; do
                    if [[ -f "$result_file" ]]; then
                        ((TOTAL_FILES++))
                        if [[ -s "$result_file" ]]; then
                            fold=$(grep "^Fold:" "$result_file" | cut -d' ' -f2)
                            test_sample=$(grep "^Test Sample:" "$result_file" | cut -d' ' -f3)
                            pcc=$(grep "^PCC:" "$result_file" | cut -d' ' -f2)
                            pvalue=$(grep "^P-value:" "$result_file" | cut -d' ' -f2)
                            nspots=$(grep "^N_spots:" "$result_file" | cut -d' ' -f2)
                            if [[ -n "$fold" && -n "$test_sample" && -n "$pcc" ]]; then
                                echo "THItoGene,$score_method,$pathway,$fold,$test_sample,$pcc,$pvalue,$nspots" >> $SUMMARY_FILE
                                ((PROCESSED++))
                            fi
                        fi
                    fi
                done
            fi
        done
    fi
done

echo "Total files: $TOTAL_FILES"
echo "Processed: $PROCESSED"
echo "Summary saved to: $SUMMARY_FILE"

if [[ $PROCESSED -gt 0 ]]; then
    echo "Results per pathway:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f3 | sort | uniq -c | sort -nr
    echo "Results per score method:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f2 | sort | uniq -c
    echo "Top 10 PCC:"
    tail -n +2 "$SUMMARY_FILE" | sort -t',' -k6 -nr | head -10 | \
        awk -F',' '{printf "%s | %s | Fold %s | PCC: %.4f\n", $2, $3, $4, $6}'
fi
