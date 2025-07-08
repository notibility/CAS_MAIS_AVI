'''
Description: unimodal encoder + concat + attention fusion for task2 (5标签输出)
'''
import torch
import torch.nn as nn
from torch.nn.functional import dropout

from .modules.encoder import MLPEncoder, LSTMEncoder

class AVIBASELINE_TASK2(nn.Module):
    def __init__(self, args):
        super(AVIBASELINE_TASK2, self).__init__()
        input_dim = 49920  # 动态传入
        output_dim = 1  # 5个标签
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            ) for _ in range(32)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        # features: [batch, 30*input_dim]
        features = features.view(features.size(0), -1)
        outputs = torch.stack([mlp(features) for mlp in self.ensemble], dim=0)
        interloss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)
        return features, outputs.mean(dim=0), interloss 