#!/bin/bash

# Summarize all HisToGene pathway prediction test results

RESULTS_DIR="./results_HisToGene_pathway"
SUMMARY_FILE="./HisToGene_pathway_performance_summary.csv"
DETAILED_SUMMARY="./HisToGene_pathway_detailed_summary.csv"
COMPARISON_FILE="./HisToGene_vs_THItoGene_comparison.csv"

echo "Summarizing HisToGene pathway prediction test results..."
echo "=================================================="

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Results directory does not exist: $RESULTS_DIR"
    echo "Run test tasks first: bash selective_test_histogene.sh"
    exit 1
fi

# Headers
echo "Model,Dataset,ScoreMethod,Pathway,PathwayType,Fold,TestSample,PCC,PValue,NSpots,PredMin,PredMax,PredMean,PredStd,GTMin,GTMax,GTMean,GTStd,TestTime" > $SUMMARY_FILE
echo "Model,Dataset,ScoreMethod,Pathway,PathwayType,Fold,TestSample,PCC,PValue,NSpots,PredMin,PredMax,PredMean,PredStd,GTMin,GTMax,GTMean,GTStd,NegPredCount,NegPredPercent,NegGTCount,NegGTPercent,TestTime,ModelPath,ResultPath" > $DETAILED_SUMMARY

echo "Collecting results..."

declare -A PATHWAY_TYPES
PATHWAY_TYPES[GO_0034340]="GO"
PATHWAY_TYPES[GO_0060337]="GO"
PATHWAY_TYPES[GO_0071357]="GO"
PATHWAY_TYPES[KEGG_P53_SIGNALING_PATHWAY]="KEGG"
PATHWAY_TYPES[KEGG_BREAST_CANCER]="KEGG"
PATHWAY_TYPES[KEGG_APOPTOSIS]="KEGG"
PATHWAY_TYPES[BREAST_CANCER_LUMINAL_A]="MSigDB"
PATHWAY_TYPES[HER2_AMPLICON_SIGNATURE]="MSigDB"
PATHWAY_TYPES[HALLMARK_MYC_TARGETS_V1]="MSigDB"

TOTAL_FILES=0
PROCESSED=0
ERRORS=0

for dataset_dir in "$RESULTS_DIR"/*; do
    if [[ -d "$dataset_dir" ]]; then
        dataset=$(basename "$dataset_dir")
        for score_dir in "$dataset_dir"/*; do
            if [[ -d "$score_dir" ]]; then
                score_method=$(basename "$score_dir")
                for pathway_dir in "$score_dir"/*; do
                    if [[ -d "$pathway_dir" ]]; then
                        pathway=$(basename "$pathway_dir")
                        pathway_type="${PATHWAY_TYPES[$pathway]:-Unknown}"
                        for result_file in "$pathway_dir"/*_results.txt; do
                            if [[ -f "$result_file" ]]; then
                                ((TOTAL_FILES++))
                                if [[ -s "$result_file" ]]; then
                                    fold=$(grep "^Fold:" "$result_file" | cut -d' ' -f2)
                                    test_sample=$(grep "^Test Sample:" "$result_file" | cut -d' ' -f3- | tr ' ' '_')
                                    pcc=$(grep "^PCC:" "$result_file" | cut -d' ' -f2)
                                    pvalue=$(grep "^P-value:" "$result_file" | cut -d' ' -f2)
                                    nspots=$(grep "^N_spots:" "$result_file" | cut -d' ' -f2)

                                    pred_range=$(grep "Prediction range:" "$result_file" | sed 's/.*\[\(.*\)\]/\1/' | tr ',' ' ')
                                    pred_mean=$(grep "Prediction mean:" "$result_file" | awk '{print $3}')
                                    pred_std=$(grep "Prediction std:" "$result_file" | awk '{print $3}')
                                    gt_range=$(grep "Ground truth range:" "$result_file" | sed 's/.*\[\(.*\)\]/\1/' | tr ',' ' ')
                                    gt_mean=$(grep "Ground truth mean:" "$result_file" | awk '{print $4}')
                                    gt_std=$(grep "Ground truth std:" "$result_file" | awk '{print $4}')
                                    neg_pred_info=$(grep "Negative values (pred):" "$result_file" | awk -F'[(/]' '{print $(NF-1)}' | tr -d '%')
                                    neg_gt_info=$(grep "Negative values (gt):" "$result_file" | awk -F'[(/]' '{print $(NF-1)}' | tr -d '%')

                                    if [[ -n "$pred_range" ]]; then
                                        pred_min=$(echo $pred_range | awk '{print $1}')
                                        pred_max=$(echo $pred_range | awk '{print $2}')
                                    else
                                        pred_min="NA"
                                        pred_max="NA"
                                    fi
                                    if [[ -n "$gt_range" ]]; then
                                        gt_min=$(echo $gt_range | awk '{print $1}')
                                        gt_max=$(echo $gt_range | awk '{print $2}')
                                    else
                                        gt_min="NA"
                                        gt_max="NA"
                                    fi

                                    pred_mean=${pred_mean:-"NA"}
                                    pred_std=${pred_std:-"NA"}
                                    gt_mean=${gt_mean:-"NA"}
                                    gt_std=${gt_std:-"NA"}
                                    neg_pred_percent=${neg_pred_info:-"NA"}
                                    neg_gt_percent=${neg_gt_info:-"NA"}
                                    test_time="NA"

                                    if [[ -n "$fold" && -n "$test_sample" && -n "$pcc" && -n "$pvalue" && -n "$nspots" ]]; then
                                        echo "HisToGene,$dataset,$score_method,$pathway,$pathway_type,$fold,$test_sample,$pcc,$pvalue,$nspots,$pred_min,$pred_max,$pred_mean,$pred_std,$gt_min,$gt_max,$gt_mean,$gt_std,$test_time" >> $SUMMARY_FILE
                                        model_path="./model/pathway/${score_method}/${pathway}/last_train_-htg_${pathway}_${score_method}_cv_${fold}.ckpt"
                                        result_path="$result_file"
                                        echo "HisToGene,$dataset,$score_method,$pathway,$pathway_type,$fold,$test_sample,$pcc,$pvalue,$nspots,$pred_min,$pred_max,$pred_mean,$pred_std,$gt_min,$gt_max,$gt_mean,$gt_std,NA,$neg_pred_percent,NA,$neg_gt_percent,$test_time,$model_path,$result_path" >> $DETAILED_SUMMARY
                                        ((PROCESSED++))
                                        echo "Processed: $dataset/$score_method/$pathway/fold_$fold (PCC=$pcc)"
                                    else
                                        echo "Skipped: $result_file (incomplete info)"
                                        ((ERRORS++))
                                    fi
                                else
                                    echo "Skipped: $result_file (empty)"
                                    ((ERRORS++))
                                fi
                            fi
                        done
                    fi
                done
            fi
        done
    fi
done

echo "=================================================="
echo "Summary completed!"
echo "Found: $TOTAL_FILES | Processed: $PROCESSED | Errors: $ERRORS"
echo "Summary file: $SUMMARY_FILE"
echo "Detailed file: $DETAILED_SUMMARY"
echo "=================================================="

if [[ $PROCESSED -gt 0 ]]; then
    echo "Quick stats:"
    echo "Results per pathway:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f4 | sort | uniq -c | sort -nr
    echo "Results per score method:"
    tail -n +2 "$SUMMARY_FILE" | cut -d',' -f3 | sort | uniq -c
    echo "Top 10 PCC:"
    tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -nr | head -10 | \
        awk -F',' '{printf "  %s | %s | %s | Fold %s | PCC: %.4f\n", $3, $4, $5, $6, $8}'
else
    echo "No valid result files found"
    echo "Check:"
    echo "  - If test tasks finished"
    echo "  - If result file format is correct"
    echo "  - If paths are correct"
fi

if [[ -f "./THItoGene_pathway_performance_summary.csv" && $PROCESSED -gt 0 ]]; then
    echo "Creating HisToGene vs THItoGene comparison..."
    echo "Pathway,PathwayType,ScoreMethod,HisToGene_Mean_PCC,HisToGene_Std_PCC,HisToGene_Count,THItoGene_Mean_PCC,THItoGene_Std_PCC,THItoGene_Count,Difference,Better_Model" > $COMPARISON_FILE
    temp_histogene="/tmp/histogene_summary.csv"
    tail -n +2 "$SUMMARY_FILE" | awk -F',' '{print $4","$5","$3","$8}' | \
        awk -F',' '{ps=$1","$2","$3; sum[ps]+=$4; count[ps]++; sumsq[ps]+=$4*$4}
        END {for(ps in sum) {mean=sum[ps]/count[ps]; if(count[ps]>1) std=sqrt((sumsq[ps]-sum[ps]*sum[ps]/count[ps])/(count[ps]-1)); else std=0; print ps","mean","std","count[ps]}}' > $temp_histogene
    temp_thitogene="/tmp/thitogene_summary.csv"
    tail -n +2 "./THItoGene_pathway_performance_summary.csv" | awk -F',' '{print $3","$2","$6}' | \
        awk -F',' '{ps=$1","$2; sum[ps]+=$3; count[ps]++; sumsq[ps]+=$3*$3}
        END {for(ps in sum) {mean=sum[ps]/count[ps]; if(count[ps]>1) std=sqrt((sumsq[ps]-sum[ps]*sum[ps]/count[ps])/(count[ps]-1)); else std=0; print ps","mean","std","count[ps]}}' > $temp_thitogene

    while IFS=',' read -r pathway pathway_type score htg_mean htg_std htg_count; do
        thi_result=$(grep "^${pathway},${score}," $temp_thitogene)
        if [[ -n "$thi_result" ]]; then
            thi_mean=$(echo $thi_result | cut -d',' -f3)
            thi_std=$(echo $thi_result | cut -d',' -f4)
            thi_count=$(echo $thi_result | cut -d',' -f5)
            difference=$(awk "BEGIN {printf \"%.4f\", $htg_mean - $thi_mean}")
            if (( $(awk "BEGIN {print ($htg_mean > $thi_mean)}") )); then
                better="HisToGene"
            else
                better="THItoGene"
            fi
            echo "$pathway,$pathway_type,$score,$htg_mean,$htg_std,$htg_count,$thi_mean,$thi_std,$thi_count,$difference,$better" >> $COMPARISON_FILE
        else
            echo "$pathway,$pathway_type,$score,$htg_mean,$htg_std,$htg_count,NA,NA,0,NA,HisToGene" >> $COMPARISON_FILE
        fi
    done < $temp_histogene
    rm -f $temp_histogene $temp_thitogene
    echo "Comparison file created: $COMPARISON_FILE"
fi

echo "Generated files:"
echo "  Summary: $SUMMARY_FILE"
echo "  Detailed: $DETAILED_SUMMARY"
[[ -f "$COMPARISON_FILE" ]] && echo "  Comparison: $COMPARISON_FILE"
echo "=================================================="
