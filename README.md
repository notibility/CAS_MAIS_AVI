# 项目使用说明

本项目支持两个赛道的多模态任务，包含文本增强、特征提取、模型训练、预测和结果合并等完整流程。以下为详细使用步骤：

---

## 1. 文本增强与特征提取

1. **文本增强**
   - 使用 `text-infer.py`，调用大模型对原始文本进行增强处理。
   - 示例命令：
     ```bash
     python text-infer.py
     ```

2. **文本拼接**
   - 使用 `merge_txts.py`，将增强后的文本与原始文本进行拼接。
   - 示例命令：
     ```bash
     python merge_txts.py
     ```

3. **特征提取**
   - 使用 `features_extract.py` 对拼接后的文本进行特征提取。
   - 示例命令：
     ```bash
     python features_extract.py
     ```

---

## 2. 赛道一（Track 1）

### 2.1 训练与预测

- **问题q3、q4、q5**
  - 推荐使用 `train_task1_merge.py` 进行训练，该脚本将原始文本和大模型增强文本融合后的文本作为文本模态进行训练。
  - 训练命令示例：
    ```bash
    python train_task1_merge.py \
        --train_csv="/AVI/train_data.csv" \
        --val_csv="/AVI/AVI_Challenge_dataset/val_data.csv" \
        --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/AVI/features/features_merged_dp_r1_qwen3" \
        --grad_clip=0.5 \
        --output_model="models/q6_model.pth" \
        --question=q6 \
        --label_col="Conscientiousness" \
        --batch_size=32 \
        --learning_rate=0.0001 \
        --num_epochs=100 \
        --hidden_dim=128 \
        --dropout=0.5 \
        --gpu 2
    ```
  - 预测命令示例：
    ```bash
    python test_avi_baseline_merge.py --audio_dir audio_feats --video_dir video_feats \
      --text_dir 融合文本特征目录 --test_csv test.csv --question q3 \
      --label_col Honesty-Humility --model_path best_q3.pth --output_csv pred_q3.csv
    # q4、q5同理
    ```

- **问题q6**
  - 推荐使用 `train_task1.py` 进行训练，该脚本将增强文本单独作为一个模态进行训练。
  - 训练命令示例：
    ```bash
    python train_task1.py \
        --train_csv="/newdisk/AVI/train_data.csv" \
        --val_csv="/AVI/avi/AVI_Challenge_dataset/val_data.csv" \
        --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/AVI/new_text_feture_2" \
        --text2_dir="/AVI/features/features_deepseek" \
        --grad_clip=0.5 \
        --output_model="models/q6_model.pth" \
        --question=q6 \
        --label_col="Conscientiousness" \
        --batch_size=32 \
        --learning_rate=0.0001 \
        --num_epochs=100 \
        --hidden_dim=128 \
        --dropout=0.5  \
        --gpu 3
    ```
  - 预测命令示例：
    ```bash
    python test_avi_baseline_merge.py \
        --test_csv="/AVI/avi/AVI_Challenge_dataset/test_data_basic_information.csv" \
        --audio_dir="AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/AVI/features/features_merged_dp_r1_qwen3" \
        --question=q3 \
        --model_path="models/q3_model_wei.pth" \
        --label_col="Honesty-Humility" \
        --output_csv=predictions_tack1_wei.csv \
        --batch_size=32 \
    python train_task1.py \
        --train_csv="/newdisk/AVI/train_data.csv" \
        --val_csv="/AVI/avi/AVI_Challenge_dataset/val_data.csv" \
        --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/AVI/new_text_feture_2" \
        --text2_dir="/AVI/features/features_deepseek" \
        --grad_clip=0.5 \
        --output_model="models/q6_model.pth" \
        --question=q6 \
        --label_col="Conscientiousness" \
        --batch_size=32 \
        --learning_rate=0.0001 \
        --num_epochs=100 \
        --hidden_dim=128 \
        --dropout=0.5  \
        --gpu 3
    ```

### 2.2 结果合并

- 使用 `merge_to_submission.py` 将各问题的预测结果合并为最终提交文件。
- 示例命令：
  ```bash
  python merge_to_submission.py
  # 该脚本会读取各问题的预测csv并生成最终提交文件 submission_track1.csv
  ```

---

## 3. 赛道二（Track 2）

- 使用 `train_task2_enhanced.py` 进行训练，模型结构见 `avi_baselin_task2.py`。
- 训练命令示例：
  ```bash
    python train_task2_enhanced.py \
        --train_csv="/data/emotion-data/AVI/avi/AVI_Challenge_dataset/train_data.csv" \
        --val_csv="/data/emotion-data/AVI/avi/AVI_Challenge_dataset/val_data.csv" \
        --audio_dir="/data/emotion-data/AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/data/emotion-data/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/newdisk/AVI/features/features_merged_dp_r1_qwen3"  \
        --output_model="task2_model_Social_Hireability.pth" \
        --question q1 q2 q3 q4 q5 q6 \
        --gpu 0 \
        --grad_clip=0.5 \
        --label_col "Integrity" "Collegiality" "Social_versatility" "Development_orientation" "Hireability"\
        --batch_size=32 \
        --learning_rate=0.0002 \
        --num_epochs=100 \
  ```
  ```bash
    python train_task2_enhanced.py \
        --train_csv="/data/emotion-data/AVI/avi/AVI_Challenge_dataset/train_data.csv" \
        --val_csv="/data/emotion-data/AVI/avi/AVI_Challenge_dataset/val_data.csv" \
        --audio_dir="/data/emotion-data/AVI/avi/AVI_future/whisper_feature" \
        --video_dir="/data/emotion-data/AVI/avi/AVI_future/video/clip_raw" \
        --text_dir="/newdisk/AVI/features/features_merged_dp_r1_qwen3"  \
        --output_model="task2_model_Social_Hireability.pth" \
        --question q1 q2 q3 q4 q5 q6 \
        --gpu 0 \
        --grad_clip=0.5 \
        --label_col "Integrity" "Collegiality" "Social_versatility" "Development_orientation" "Hireability"\
        --batch_size=32 \
        --learning_rate=0.0002 \
        --num_epochs=100 \
  ```

---

## 4. 其他说明

- 请根据实际数据路径和特征目录调整命令参数。
- 训练和预测脚本均支持GPU加速，可通过`--gpu`参数指定GPU编号。
- 详细参数说明请参考各脚本内的注释。

---

如有问题请联系项目维护者yanglongjiang2024@ia.ac.cn