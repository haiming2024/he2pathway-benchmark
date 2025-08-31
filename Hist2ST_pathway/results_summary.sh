#!/bin/bash

RESULTS_DIR="./results"
SUMMARY_FILE="./pathway_performance_summary.csv"

echo "Summarizing Hist2ST test results..."

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Results directory does not exist: $RESULTS_DIR"
    exit 1
fi

echo "ScoreMethod,Pathway,Fold,TestSample,PCC,PValue,NSpots" > $SUMMARY_FILE

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
                            
                            if [[ -n "$fold" && -n "$test_sample" && -n "$pcc" && -n "$pvalue" && -n "$nspots" ]]; then
                                echo "$score_method,$pathway,$fold,$test_sample,$pcc,$pvalue,$nspots" >> $SUMMARY_FILE
                                ((PROCESSED++))
                                echo "Processed: $score_method/$pathway/fold_$fold"
                            else
                                echo "Skipped: $result_file (incomplete information)"
                            fi
                        else
                            echo "Skipped: $result_file (empty file)"
                        fi
                    fi
                done
            fi
        done
    fi
done

echo "Summary finished!"
echo "Found: $TOTAL_FILES, Processed: $PROCESSED"
echo "Output: $SUMMARY_FILE"

if [[ $PROCESSED -gt 0 ]]; then
    echo "Quick stats:"
    echo "Results per pathway:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f2 | sort | uniq -c | sort -nr
    
    echo "Results per scoring method:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f1 | sort | uniq -c
    
    echo "Top 10 PCC:"
    tail -n +2 "$SUMMARY_FILE" | sort -t',' -k5 -nr | head -10 | \
        awk -F',' '{printf "  %s | %s | Fold %s | PCC: %.4f\n", $1, $2, $3, $5}'
    
    echo "For further analysis with Python:"
    echo "  import pandas as pd"
    echo "  df = pd.read_csv('$SUMMARY_FILE')"
    echo "  df.groupby(['ScoreMethod', 'Pathway'])['PCC'].agg(['mean', 'std', 'count'])"
else
    echo "No valid result files found."
fi
