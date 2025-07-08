import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.avi_task1_merge import AVIBASELINE_TWO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question

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
            audio_features = np.expand_dims(audio_features, 0)  # [1, audio_dim]
        elif len(audio_features.shape) == 3:
            audio_features = np.mean(audio_features, axis=0)  # [T, audio_dim]
            if audio_features.shape[0] > 5:
                step = audio_features.shape[0] // 5
                audio_features = np.array([np.mean(audio_features[i:i+step], axis=0) for i in range(0, audio_features.shape[0], step)][:5])
            elif audio_features.shape[0] < 5:
                audio_features = np.tile(audio_features, (5 // audio_features.shape[0] + 1, 1))[:5]

        video_features = np.tile(np.expand_dims(video_features, 0), (5, 1))
        text_features = np.tile(np.expand_dims(text_features, 0), (5, 1))

        batch = {
            'audios': torch.tensor(audio_features, dtype=torch.float32),  # [5, audio_dim]
            'texts': torch.tensor(text_features, dtype=torch.float32),    # [5, text_dim]
            'videos': torch.tensor(video_features, dtype=torch.float32),  # [5, video_dim]
            'id': sample_id
        }
        return batch

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, col, question=None):
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['id']
            input_batch = {k: v.to(device) for k, v in batch.items() if k != 'id'}
            _, outputs, _ = model(input_batch)
            if outputs.dim() > 1:
                outputs = outputs.mean(dim=1)
            predictions.extend((outputs.squeeze().cpu().numpy() * 4 + 1))
            sample_ids.extend(ids)

    results = pd.DataFrame({'id': sample_ids, col: predictions})
    if question is not None and question != '':
        base, ext = os.path.splitext(output_csv)
        output_csv = f"{base}_q{question}{ext}"
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    test_dataset = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: {
        'audios': torch.stack([item['audios'] for item in x]),
        'videos': torch.stack([item['videos'] for item in x]),
        'texts': torch.stack([item['texts'] for item in x]),
        'id': [item['id'] for item in x]
    })

    # 你需要根据训练时的参数设置这些维度
    class Args:
        text_dim = 768
        audio_dim = 384
        video_dim = 512
        dropout = 0.1
        hidden_dim = 128
        grad_clip = 1.0
    model_args = Args()
    model = AVIBASELINE_TWO(model_args).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_col, args.question)

if __name__ == '__main__':
    main() 