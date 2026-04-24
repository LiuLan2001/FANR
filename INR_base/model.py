import torch
import torch.nn as nn
import math
from math import exp, log
import tinycudann as tcnn
import numpy as np 

class NGP_TCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_input_dims = 4
        n_output_dims = 1

        n_levels = 13
        n_features_per_level = 2
        log2_hashmap_size = 15
        base_resolution = 16
        max_resolution = 256

        per_level_scale = exp(
            (log(max_resolution) - log(base_resolution)) / (n_levels - 1)
        )
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "CutlassMLP",  
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        y = self.model(x)
        return y.float()


# coordnet
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
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features):
        super(ResBlock,self).__init__()
        self.sine1 = SineLayer(in_features, out_features)
        self.sine2 = SineLayer(out_features, out_features)
        self.flag = (in_features!=out_features)
        if self.flag:
            self.transform = SineLayer(in_features,out_features)
    
    def forward(self,features):
        outputs = self.sine1(features)
        outputs = self.sine2(outputs)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)

class CoordNet(nn.Module):
    def __init__(self, in_features, out_features, init_features=64,num_res = 10):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(ResBlock(in_features,init_features))
        self.net.append(ResBlock(init_features,2*init_features))
        self.net.append(ResBlock(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))
        self.net.append(ResBlock(4*init_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
    
# Finer
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, bias_range=1.0):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights(bias_range)

    def init_weights(self, bias_range):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                     np.sqrt(6 / self.in_features) / self.omega_0
                )

            self.linear.bias.uniform_(-bias_range, bias_range)

    def forward(self, x):
        y = self.linear(x)
        return torch.sin(self.omega_0 * (torch.abs(y) + 1.0) * y)
    
class Finer(nn.Module):
    def __init__(self,
                 in_features=4,
                 hidden_features=256,
                 hidden_layers=13,
                 out_features=1,
                 omega_0=10,
                 bias_range=0.1):
        super().__init__()

        layers = []

        layers.append(FinerLayer(in_features, hidden_features,
                                 is_first=True,
                                 omega_0=omega_0,
                                 bias_range=bias_range))
        for _ in range(hidden_layers):
            layers.append(FinerLayer(hidden_features, hidden_features,
                                     is_first=False,
                                     omega_0=omega_0,
                                     bias_range=bias_range))
        final_linear = nn.Linear(hidden_features, out_features)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
    
#kdinr
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
        
    def forward(self, input):
        return self.linear(input)

class BottleNeckBlock(nn.Module):
    def __init__(self,in_features):
        super(BottleNeckBlock,self).__init__()
        self.net = []
        self.net.append(SineLayer(in_features, in_features//4))
        self.net.append(SineLayer(in_features//4, in_features//4))
        self.net.append(SineLayer(in_features//4, in_features))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, features):
        outputs = self.net(features)
        return 0.5 * (outputs+features)
    
class KDINR(nn.Module):
    def __init__(self, in_features, out_features, init_features=256,num_res=1):
        super(KDINR,self).__init__()

        self.num_res = num_res

        self.net = []
        self.net.append(SineLayer(in_features,init_features))
        self.net.append(SineLayer(init_features,2*init_features))
        self.net.append(SineLayer(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(BottleNeckBlock(4*init_features))
        self.net.append(LinearLayer(4*init_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output