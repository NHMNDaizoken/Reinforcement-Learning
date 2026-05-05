#!/bin/bash
# nohup ./run_all_train.sh > /dev/null 2>&1 &
# tail -f train_results.log
# Khai báo file lưu log
LOG_FILE="train_results.log"

echo "=============================================" | tee $LOG_FILE
echo "🚀 BẮT ĐẦU CHUỖI HUẤN LUYỆN (TRAIN ALL)" | tee -a $LOG_FILE
echo "Thời gian bắt đầu: $(date)" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

# ---------------------------------------------------------
# PHẦN 1: TRAIN MÔ HÌNH DQN ĐƠN LẺ
# ---------------------------------------------------------
echo -e "\n\n=============================================" | tee -a $LOG_FILE
echo "🧠 PHẦN 1: TRAIN DQN ĐƠN LẺ (500 Episodes)" | tee -a $LOG_FILE
echo "Môi trường: Chỉ học trực tiếp trên 3x3 Gaussian" | tee -a $LOG_FILE
echo "File lưu model: models/best_single.pth" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

time python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flow configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --episodes 500 \
  --model-path models/best_single.pth 2>&1 | tee -a $LOG_FILE


# ---------------------------------------------------------
# PHẦN 2: TRAIN MÔ HÌNH DQN CURRICULUM
# ---------------------------------------------------------
echo -e "\n\n=============================================" | tee -a $LOG_FILE
echo "🎓 PHẦN 2: TRAIN DQN CURRICULUM (1200 Episodes)" | tee -a $LOG_FILE
echo "Lộ trình: 6 Giai đoạn (từ đường vắng đến kẹt xe nặng)" | tee -a $LOG_FILE
echo "File lưu model: models/best_curriculum.pth" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

time python src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/flow_low_flat.json configs/flow_low_peak.json configs/flow_medium_flat.json configs/flow_medium_peak.json configs/flow_high_flat.json configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode curriculum \
  --curriculum-interval 200 \
  --episodes 1200 \
  --model-path models/best_curriculum.pth 2>&1 | tee -a $LOG_FILE


# ---------------------------------------------------------
echo -e "\n=============================================" | tee -a $LOG_FILE
echo "✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!" | tee -a $LOG_FILE
echo "Thời gian kết thúc: $(date)" | tee -a $LOG_FILE
echo "Vui lòng chạy file run_all_evals.sh để đánh giá kết quả." | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

