'''
Description: unimodal encoder + concat + attention fusion
'''
import torch
import torch.nn as nn
import random
import numpy as np
from torch.nn.functional import dropout

from .modules.encoder import MLPEncoder, LSTMEncoder

def set_model_seed(seed):
    """设置模型相关的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AVIBASELINE_TWO(nn.Module):
    def __init__(self, args):
        super(AVIBASELINE_TWO, self).__init__()
        input_dim = 24960 # 8320 + 768 (新增的文本特征维度)
        output_dim = 1
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        # 修改为4个模态
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            ) for _ in range(16)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        audio_features = batch['audios']
        text_features = batch['texts']
        video_features = batch['videos']

        features = torch.cat([audio_features, video_features, text_features], dim=-1)

        features = features.view(features.size(0), -1)  # 将特征展平
        outputs = torch.stack([mlp(features) for mlp in self.ensemble], dim=0)
        interloss = torch.tensor(0.0).cuda()

        return features, outputs.mean(dim=0), interloss
