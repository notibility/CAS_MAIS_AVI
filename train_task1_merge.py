import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from models.avi_task1_merge import AVIBASELINE_TWO
from scipy.interpolate import interp1d

def get_feature_shape(feature_file):
    """获取特征文件的完整形状"""
    feature = np.load(feature_file)
    if len(feature.shape) == 1:
        return (1, feature.shape[0])  # 将一维特征转换为二维
    return feature.shape

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question
        self.label_col = label_col

        # 获取第一个样本的特征维度
        sample_id = self.data.iloc[0]['id']
        
        # 获取文本特征维度
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{self.question}")][0]
        text_shape = get_feature_shape(os.path.join(self.text_dir, text_file))
        self.text_dim = text_shape[1]
        print(f"Text feature shape: {text_shape}, dimension: {self.text_dim}")
        
        # 获取音频特征维度
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{self.question}")][0]
        audio_shape = get_feature_shape(os.path.join(self.audio_dir, audio_file))
        self.audio_dim = audio_shape[1]
        print(f"Audio feature shape: {audio_shape}, dimension: {self.audio_dim}")
        
        # 获取视频特征维度
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{self.question}")][0]
        video_shape = get_feature_shape(os.path.join(self.video_dir, video_file))
        self.video_dim = video_shape[1]
        print(f"Video feature shape: {video_shape}, dimension: {self.video_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{self.question}")]
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{self.question}")]
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{self.question}")]

        if len(audio_file) == 0 or len(video_file) == 0 or len(text_file) == 0:
            raise FileNotFoundError(f"Files for {sample_id}_{self.question} not found.")

        audio_features = np.load(os.path.join(self.audio_dir, audio_file[0]))
        video_features = np.load(os.path.join(self.video_dir, video_file[0]))
        text_features = np.load(os.path.join(self.text_dir, text_file[0]))

        # 处理音频特征
        if len(audio_features.shape) == 1:
            audio_features = np.expand_dims(audio_features, 0)  # [1, 1280]
        elif len(audio_features.shape) == 3:
            audio_features = np.mean(audio_features, axis=0)  # [T, 1280]
            if audio_features.shape[0] > 5:
                step = audio_features.shape[0] // 5
                audio_features = np.array([np.mean(audio_features[i:i+step], axis=0) for i in range(0, audio_features.shape[0], step)][:5])
            elif audio_features.shape[0] < 5:
                audio_features = np.tile(audio_features, (5 // audio_features.shape[0] + 1, 1))[:5]
        
        video_features = np.tile(np.expand_dims(video_features,0),(5,1))
        text_features = np.tile(np.expand_dims(text_features, 0), (5, 1))
        
        # 准备模型输入格式
        batch = {
            'audios': torch.tensor(audio_features, dtype=torch.float32),  # [5, 1280] 或 [1, 1280]
            'texts': torch.tensor(text_features, dtype=torch.float32),    # [5, 768]
            'videos': torch.tensor(video_features, dtype=torch.float32)   # [5, 512]
        }
        
        label = self.data.iloc[idx][self.label_col]
        label_normalized = (label - 1) / 4

        return batch, torch.tensor(label_normalized, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch, labels in train_loader:
        # 将数据移动到设备
        for key in batch:
            batch[key] = batch[key].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        features, outputs, _ = model(batch)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, predictions, targets = 0, [], []
    with torch.no_grad():
        for batch, labels in loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            labels = labels.to(device)
            
            features, outputs, _ = model(batch)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            predictions.append(outputs.squeeze().cpu().numpy())
            targets.append(labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # 计算归一化后的MSE
    mse_normalized = mean_squared_error(targets, predictions)
    
    # 转换回原始尺度（1-5分制）
    predictions_original = predictions * 4 + 1
    targets_original = targets * 4 + 1
    
    # 计算原始尺度的MSE和平均误差
    mse_original = mean_squared_error(targets_original, predictions_original)
    mean_error = np.mean(np.abs(targets_original - predictions_original))
    
    print(f"归一化后的MSE: {mse_normalized:.4f}")
    print(f"原始尺度的MSE: {mse_original:.4f}")
    print(f"原始尺度的平均误差: {mean_error:.4f}分")
    
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
    parser.add_argument('--question', required=True)
    parser.add_argument('--label_col', default='Hireability')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--output_model', default='best_model.pth')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的GPU编号，如0或0,1')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集
    train_set = MultimodalDataset(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col)
    val_set = MultimodalDataset(args.val_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col)

    # 设置特征维度
    args.text_dim = train_set.text_dim
    args.audio_dim = train_set.audio_dim if hasattr(train_set, 'audio_dim') else 384
    args.video_dim = train_set.video_dim if hasattr(train_set, 'video_dim') else 512
    args.feat_type = 'frm_align'

    print(f"Model parameters:")
    print(f"- Text dimension: {args.text_dim}")
    print(f"- Audio dimension: {args.audio_dim}")
    print(f"- Video dimension: {args.video_dim}")
    print(f"- Hidden dimension: {args.hidden_dim}")
    print(f"- Feature type: {args.feat_type}")
    print(f"- Total input dimension: {args.text_dim * 2 + args.audio_dim + args.video_dim}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = AVIBASELINE_TWO(args).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_model)
            best_model_path = args.output_model
            
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}")

    print('best_val_loss:', best_val_loss)
    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

if __name__ == '__main__':
    main() 