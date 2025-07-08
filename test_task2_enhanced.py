import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models.avi_task2 import AVIBASELINE_TASK2

def get_feature_shape(feature_file):
    feature = np.load(feature_file)
    if len(feature.shape) == 1:
        return (1, feature.shape[0])
    return feature.shape

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, questions, label_cols, rating_csv=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.questions = questions
        self.label_cols = label_cols
        if rating_csv is not None:
            self.rating = pd.read_csv(rating_csv)
            self.result_dict = {row['id']: row for _, row in self.rating.iterrows()}
        else:
            self.rating = None
            self.result_dict = None

        # 获取第一个样本的特征维度
        sample_id = self.data.iloc[0]['id']
        q = self.questions[0]
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")][0]
        text_shape = get_feature_shape(os.path.join(self.text_dir, text_file))
        self.text_dim = text_shape[1]
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")][0]
        audio_shape = get_feature_shape(os.path.join(self.audio_dir, audio_file))
        self.audio_dim = audio_shape[1]
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")][0]
        video_shape = get_feature_shape(os.path.join(self.video_dir, video_file))
        self.video_dim = video_shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_files = []
        video_files = []
        text_files = []
        for q in self.questions:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]
            if not audio_file or not video_file or not text_file:
                raise FileNotFoundError(f"Missing modality for {sample_id}_{q}")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])
        # 处理音频特征：先对每个文件的第二个维度取平均值，再拼接
        audio_features = np.concatenate([np.mean(np.load(os.path.join(self.audio_dir, f)), axis=0) for f in audio_files], axis=0)
        video_features = np.tile(np.concatenate([
            np.expand_dims(np.load(os.path.join(self.video_dir, f)), axis=0) for f in video_files
        ], axis=0), (5, 1))
        text_features = np.tile(np.concatenate([
            np.expand_dims(np.load(os.path.join(self.text_dir, f)), axis=0) for f in text_files
        ], axis=0), (5, 1))
        features = np.concatenate([audio_features, video_features, text_features], axis=-1)
        return torch.tensor(features, dtype=torch.float32), sample_id

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, label_cols, device):
    predictions, sample_ids = [], []
    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)
            _, outputs, _ = model(features)
            predictions.extend((outputs.cpu().numpy() * 4 + 1))
            sample_ids.extend(ids)
    predictions = np.array(predictions)
    results = pd.DataFrame(data=predictions, columns=label_cols)
    results.insert(0, 'id', sample_ids)
    results.to_csv(output_csv, index=False)
    print(f"预测结果已保存到 {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--questions', nargs='+', required=True, help='6个问题编号')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_cols', nargs='+', required=True, help='5个标签名')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions, args.label_cols)
    input_dim = tmp.text_dim * len(args.questions) + tmp.audio_dim * len(args.questions) + tmp.video_dim * len(args.questions)
    args.input_dim = input_dim

    test_set = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions, args.label_cols)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = AVIBASELINE_TASK2(args).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_cols, device)

if __name__ == '__main__':
    main()
