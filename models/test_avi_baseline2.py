import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from train_task2 import DeepEnsembleMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, questions):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.questions = questions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_features, video_features, text_features = [], [], []

        for q in self.questions:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]
            if not audio_file or not video_file or not text_file:
                raise FileNotFoundError(f"Missing file for {sample_id}_{q}")
            audio_features.append(np.mean(np.load(os.path.join(self.audio_dir, audio_file[0])), axis=0))
            video_features.append(np.mean(np.load(os.path.join(self.video_dir, video_file[0])), axis=0))
            text_features.append(np.mean(np.load(os.path.join(self.text_dir, text_file[0])), axis=0))

        audio_features = np.concatenate(audio_features, axis=0)
        video_features = np.concatenate(video_features, axis=0)
        text_features = np.concatenate(text_features, axis=0)
        features = np.concatenate([audio_features, video_features, text_features], axis=0)
        return torch.tensor(features, dtype=torch.float32), sample_id

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, label_cols):
    predictions, sample_ids = [], []
    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)
            outputs = model(features)
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

    args = parser.parse_args()

    # 自动推断特征维度
    tmp = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions)
    input_dim = tmp[0][0].shape[0]
    del tmp

    test_set = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = DeepEnsembleMLP(input_dim, len(args.label_cols), num_mlps=32).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_cols)

if __name__ == '__main__':
    main()