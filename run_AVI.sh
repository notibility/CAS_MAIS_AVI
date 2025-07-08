# # # 极简训练脚本（一次性执行，无参数覆盖功能）Extraversion Honesty-Humility Agreeableness Conscientiousness
# python train_task1.py \
#     --train_csv="/newdisk/AVI/train_data.csv" \
#     --val_csv="/AVI/avi/AVI_Challenge_dataset/val_data.csv" \
#     --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
#     --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
#     --text_dir="/AVI/new_text_feture_2" \
#     --text2_dir="/AVI/features/features_deepseek" \
#     --grad_clip=0.5 \
#     --output_model="models/q6_model.pth" \
#     --question=q6 \
#     --label_col="Conscientiousness" \
#     --batch_size=32 \
#     --learning_rate=0.0001 \
#     --num_epochs=100 \
#     --hidden_dim=128 \
#     --dropout=0.5  \
#     --gpu 3
# # /newdisk/AVI/new_transcripts_two_features
# # /newdisk/AVI/new_text_feture_2
# # /newdisk/AVI/features/features_deepseek
# python train_task1_merge.py \
#     --train_csv="/AVI/train_data.csv" \
#     --val_csv="/AVI/AVI_Challenge_dataset/val_data.csv" \
#     --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
#     --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
#     --text_dir="/AVI/features/features_merged_dp_r1_qwen3" \
#     --grad_clip=0.5 \
#     --output_model="models/q6_model.pth" \
#     --question=q6 \
#     --label_col="Conscientiousness" \
#     --batch_size=32 \
#     --learning_rate=0.0001 \
#     --num_epochs=100 \
#     --hidden_dim=128 \
#     --dropout=0.5 \
#     --gpu 2

python train_task2.py \
    --train_csv="/AVI/avi/AVI_Challenge_dataset/train_data.csv" \
    --val_csv="/AVI_Challenge_dataset/val_data.csv" \
    --audio_dir="/AVI_future/whisper_feature" \
    --video_dir="/AVI_future/video/clip_raw" \
    --text_dir="/AVI/features/features_merged_dp_r1_qwen3"  \
    --output_model="task2_model.pth" \
    --question q1 q2 q3 q4 q5 q6 \
    --gpu 0 \
    --grad_clip=0.5 \
    --label_col "Integrity" "Collegiality" "Social_versatility" "Development_orientation" "Hireability"\
    --batch_size=32 \
    --learning_rate=0.0002 \
    --num_epochs=100 \