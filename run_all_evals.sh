#!/bin/bash
# Hướng dẫn chạy ngầm:
# chmod +x run_all_evals.sh
# nohup ./run_all_evals.sh > /dev/null 2>&1 &
# tail -f eval_results.log

# Khai báo đường dẫn bản đồ và GỘP 2 model vào 1 biến
MODELS="models/best_single.pth models/best_curriculum.pth"
ROADNET="configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json"
GAUSSIAN="configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json"

LOG_FILE="eval_results.log"

echo "=============================================" | tee $LOG_FILE
echo "🚀 BẮT ĐẦU CHẠY FULL EVALUATION (ĐÁNH GIÁ TỔNG HỢP 3 PHƯƠNG PHÁP)" | tee -a $LOG_FILE
echo "Thời gian bắt đầu: $(date)" | tee -a $LOG_FILE
echo "Các Models sẽ test chung: $MODELS" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

echo -e "\n[1/4] Đang đánh giá trên môi trường: 3x3 Gaussian..." | tee -a $LOG_FILE
time CUDA_VISIBLE_DEVICES="" python src/phase3_eval/evaluate.py --roadnet $ROADNET --flow $GAUSSIAN --models $MODELS --episodes 50 2>&1 | tee -a $LOG_FILE

echo -e "\n[2/4] Đang đánh giá trên môi trường: High Flat..." | tee -a $LOG_FILE
time CUDA_VISIBLE_DEVICES="" python src/phase3_eval/evaluate.py --roadnet $ROADNET --flow configs/flow_high_flat.json --models $MODELS --episodes 50 2>&1 | tee -a $LOG_FILE

echo -e "\n[3/4] Đang đánh giá trên môi trường: Low Peak..." | tee -a $LOG_FILE
time CUDA_VISIBLE_DEVICES="" python src/phase3_eval/evaluate.py --roadnet $ROADNET --flow configs/flow_low_peak.json --models $MODELS --episodes 50 2>&1 | tee -a $LOG_FILE

echo -e "\n[4/4] Đang đánh giá trên môi trường: High Peak..." | tee -a $LOG_FILE
time CUDA_VISIBLE_DEVICES="" python src/phase3_eval/evaluate.py --roadnet $ROADNET --flow configs/flow_high_peak.json --models $MODELS --episodes 50 2>&1 | tee -a $LOG_FILE

echo -e "\n=============================================" | tee -a $LOG_FILE
echo "✅ HOÀN THÀNH TOÀN BỘ 4 BÀI ĐÁNH GIÁ TỔNG HỢP!" | tee -a $LOG_FILE
echo "Thời gian kết thúc: $(date)" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE