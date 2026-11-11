#!/bin/bash
# Run all OSS model experiments (Step 4a + 5c) for DeepSeek R1 and Qwen 3 32B
# Run: midway auth && bash experiments/oss_models/run_all_oss.sh

set -e

echo "============================================"
echo "Step 4a: Qwen 3 32B (SQL agent)"
echo "============================================"
python3 experiments/oss_models/step4a/run_step4a_oss.py \
    --model qwen.qwen3-32b-v1:0 \
    --output experiments/oss_models/step4a/results_qwen3.tsv

echo ""
echo "============================================"
echo "Step 4a: DeepSeek R1 (SQL agent)"
echo "============================================"
python3 experiments/oss_models/step4a/run_step4a_oss.py \
    --model us.deepseek.r1-v1:0 \
    --output experiments/oss_models/step4a/results_deepseek.tsv

echo ""
echo "============================================"
echo "Step 5c: Qwen 3 32B (PM agent + examples)"
echo "============================================"
python3 experiments/oss_models/step5c/run_step5c_oss.py \
    --model qwen.qwen3-32b-v1:0 \
    --output experiments/oss_models/step5c/results_qwen3.tsv

echo ""
echo "============================================"
echo "Step 5c: DeepSeek R1 (PM agent + examples)"
echo "============================================"
python3 experiments/oss_models/step5c/run_step5c_oss.py \
    --model us.deepseek.r1-v1:0 \
    --output experiments/oss_models/step5c/results_deepseek.tsv

echo ""
echo "============================================"
echo "ALL RUNS COMPLETE. Now evaluating..."
echo "============================================"

echo ""
echo "--- Step 4a: Qwen 3 32B ---"
python3 experiments/evaluation/eval_quick_colagnostic.py \
    experiments/oss_models/step4a/results_qwen3.tsv \
    --exclude-groups 1,23,5,109,132

echo ""
echo "--- Step 4a: DeepSeek R1 ---"
python3 experiments/evaluation/eval_quick_colagnostic.py \
    experiments/oss_models/step4a/results_deepseek.tsv \
    --exclude-groups 1,23,5,109,132

echo ""
echo "--- Step 5c: Qwen 3 32B ---"
python3 experiments/evaluation/eval_quick_colagnostic.py \
    experiments/oss_models/step5c/results_qwen3.tsv \
    --exclude-groups 1,23,5,109,132

echo ""
echo "--- Step 5c: DeepSeek R1 ---"
python3 experiments/evaluation/eval_quick_colagnostic.py \
    experiments/oss_models/step5c/results_deepseek.tsv \
    --exclude-groups 1,23,5,109,132

echo ""
echo "============================================"
echo "ALL DONE!"
echo "============================================"
