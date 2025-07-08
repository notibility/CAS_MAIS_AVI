# python test_avi_baseline.py \
#     --test_csv="/AVI/avi/AVI_Challenge_dataset/test_data_basic_information.csv" \
#     --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
#     --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
#     --text_dir="/AVI/new_text_feture_2" \
#     --text2_dir="/AVI/features/features_deepseek" \
#     --question=q6 \
#     --model_path="models/q6_model_wei.pth" \
#     --label_col="Conscientiousness" \
#     --output_csv=predictions_tack1_wei.csv \
#     --batch_size=32 \
# python test_avi_baseline_merge.py \
#     --test_csv="/AVI/avi/AVI_Challenge_dataset/test_data_basic_information.csv" \
#     --audio_dir="AVI/avi/AVI_future/whisper_feature" \
#     --video_dir="/AVI/avi/AVI_future/video/clip_raw" \
#     --text_dir="/AVI/features/features_merged_dp_r1_qwen3" \
#     --question=q3 \
#     --model_path="models/q3_model_wei.pth" \
#     --label_col="Honesty-Humility" \
#     --output_csv=predictions_tack1_wei.csv \
#     --batch_size=32 \
# python test_task2_enhanced.py \
#     --test_csv="/AVI/avi/AVI_Challenge_dataset/test_data_basic_information_track2.csv" \
#     --audio_dir="/AVI/avi/AVI_future/whisper_feature" \
#     --video_dir="AVI/avi/AVI_future/video/clip_raw" \
#     --text_dir="/AVI/features/features_merged_dp_r1_qwen3" \
#     --model_path="task2_model.pth" \
#     --question q1 q2 q3 q4 q5 q6 \
#     --label_col "Integrity" "Collegiality" "Social_versatility" "Development_orientation" "Hireability"\
#     --output_csv="submission_track2.csv" \
#     --batch_size=32 \