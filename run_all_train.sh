#!/bin/bash
# nohup ./run_all_train.sh > /dev/null 2>&1 &
# tail -f logs/train_gaussian.log
PYTHON="/Users/admin/CityFlow/.venv/bin/python"
LOG_FILE="logs/train_gaussian.log"
mkdir -p logs

echo "=============================================" | tee $LOG_FILE
echo "BẮT DAU TRAIN DQN - GAUSSIAN 3x3" | tee -a $LOG_FILE
echo "Episodes: 500 | Flow: syn_3x3_gaussian_500_1h" | tee -a $LOG_FILE
echo "Thoi gian bat dau: $(date)" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

time PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" $PYTHON -u src/phase2_dqn/train_dqn.py \
  --roadnet configs/syn_3x3_gaussian_500_1h/roadnet_3X3.json \
  --flows configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json \
  --mode single \
  --episodes 500 \
  --model-path models/best.pth 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE
echo "HOAN TAT TRAINING!" | tee -a $LOG_FILE
echo "Thoi gian ket thuc: $(date)" | tee -a $LOG_FILE
echo "Chay run_all_evals.sh de danh gia ket qua." | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE
