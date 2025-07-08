import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
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
        if self.result_dict is not None:
            labels = np.array([(self.result_dict[sample_id][col] - 1) / 4 for col in self.label_cols])
        else:
            labels = self.data.iloc[idx][self.label_cols].values.astype(np.float32)
            labels = (labels - 1) / 4
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        _, outputs, _ = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, predictions, targets = 0, [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            _, outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    mse_normalized = mean_squared_error(targets, predictions)
    predictions_original = predictions * 4 + 1
    targets_original = targets * 4 + 1
    mse_original = mean_squared_error(targets_original, predictions_original)
    mean_error = np.mean(np.abs(targets_original - predictions_original))
    print(f"归一化MSE: {mse_normalized:.4f} | 原始MSE: {mse_original:.4f} | 平均误差: {mean_error:.4f}")
    return total_loss / len(loader), mse_normalized

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--questions', nargs='+', required=True, help='6个问题编号')
    parser.add_argument('--label_cols', nargs='+', required=True, help='5个标签名')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--output_model', default='task2_best_model.pth')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp = MultimodalDataset(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions, args.label_cols)
    input_dim = tmp.text_dim * len(args.questions) + tmp.audio_dim * len(args.questions) + tmp.video_dim * len(args.questions)
    args.input_dim = input_dim

    train_set = MultimodalDataset(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions, args.label_cols)
    val_set = MultimodalDataset(args.val_csv, args.audio_dir, args.video_dir, args.text_dir, args.questions, args.label_cols)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = AVIBASELINE_TASK2(args).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_model)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}")

    print('训练完成，最优模型已保存。')

if __name__ == '__main__':
    main()