import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, 
                     np.sqrt(6 / self.in_features) / self.omega_0
                )
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(
                -1 / self.in_features,
                 1 / self.in_features
            )

    def forward(self, input):
        return self.linear(input)


class BottleNeckBlock(nn.Module):
    def __init__(self, in_features):
        super(BottleNeckBlock, self).__init__()
        hidden = in_features // 4
        self.net = nn.Sequential(
            SineLayer(in_features, hidden),
            SineLayer(hidden, hidden),
            SineLayer(hidden, in_features)
        )

    def forward(self, features):
        outputs = self.net(features)
        return 0.5 * (outputs + features)


class  FINR(nn.Module):
    def __init__(self, in_features=4, out_features=1, init_features=256, num_res=1):
        super(FINR, self).__init__()

        hidden_dim = 4 * init_features
        self.hidden_dim = hidden_dim

        trunk = []
        trunk.append(SineLayer(in_features, init_features, is_first=True))
        trunk.append(SineLayer(init_features, 2 * init_features))
        trunk.append(SineLayer(2 * init_features, hidden_dim))
        for _ in range(num_res):
            trunk.append(BottleNeckBlock(hidden_dim))
        self.trunk = nn.Sequential(*trunk)

        self.low_gate = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.high_gate = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        # self.alpha_high = nn.Parameter(torch.full((hidden_dim,), 0.0))

        self.low_head = LinearLayer(hidden_dim, out_features)
        self.high_head = LinearLayer(hidden_dim, out_features)

    def forward(self, coords):
        feat = self.trunk(coords)              

        low_mask = self.low_gate(coords)       
        high_mask = self.high_gate(coords)     

        low_feat = feat * low_mask
        high_feat = feat * high_mask

        mixed_low = low_feat
        mixed_high = high_feat #+ self.alpha_high.unsqueeze(0) * low_feat

        low_pred = self.low_head(mixed_low)
        high_pred = self.high_head(mixed_high)
        final_pred = low_pred + high_pred
        return {
            "final": final_pred,
            "low": low_pred,
            "high": high_pred,
            "low_mask": low_mask,
            "high_mask": high_mask,
        }

class FINR_SS(nn.Module):
    def __init__(self, in_features=4, out_features=1, init_features=256, num_res=1):
        super(FINR_SS, self).__init__()

        hidden_dim = 4 * init_features
        self.hidden_dim = hidden_dim

        trunk1 = []
        trunk1.append(SineLayer(in_features, init_features, is_first=True))
        trunk1.append(SineLayer(init_features, 2 * init_features))
        trunk1.append(SineLayer(2 * init_features, hidden_dim))
        for _ in range(num_res):
            trunk1.append(BottleNeckBlock(hidden_dim))
        self.trunk1 = nn.Sequential(*trunk1)

        trunk2 = []
        trunk2.append(SineLayer(in_features, init_features, is_first=True))
        trunk2.append(SineLayer(init_features, 2 * init_features))
        trunk2.append(SineLayer(2 * init_features, hidden_dim))
        for _ in range(num_res):
            trunk2.append(BottleNeckBlock(hidden_dim))
        self.trunk2 = nn.Sequential(*trunk2)

        self.gate1 = nn.Linear(in_features, 2, bias=False)
        self.gate2 = nn.Linear(in_features, 2, bias=False)

        self.low_head = LinearLayer(hidden_dim, out_features)
        self.high_head = LinearLayer(hidden_dim, out_features)

    def forward(self, coords):
        out1 = self.trunk1(coords) 
        out2 = self.trunk2(coords)    

        g1 = F.softmax(self.gate1(coords), dim=1)
        g2 = F.softmax(self.gate2(coords), dim=1)           

        low_pred = self.low_head(g1[:, 0:1] * out1 + g1[:, 1:2] * out2)
        high_pred = self.high_head(g2[:, 0:1] * out1 + g2[:, 1:2] * out2)
        final_pred = low_pred + high_pred
        return {
            "final": final_pred,
            "low": low_pred,
            "high": high_pred,
        }