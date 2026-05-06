import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from utils import (BatchIndex, get_mgrid, fast_random_choice, cleanup, dataset_selection, adjust_lr,)
from .load_data import LoadData
import numpy as np
import torch
import os
import queue
import tqdm
import torch.nn.functional as F   
from concurrent.futures import ThreadPoolExecutor

class ScalarDataSet(LoadData):
    def __init__(self, args, device='cuda:0'):
        self.dataset, self.batch_size = args.dataset, args.batch_size
        self.temporal, self.spatial = args.temporal, args.spatial
        self.device = device   
        self.gaussian_sigma = getattr(args, "gaussian_sigma", 1.0)  # [MOD]

        self.ori_dim, self.total_samples, self.data_path, self.downsample_factor = dataset_selection(
            self.dataset, self.spatial, self.temporal
        )
        self.dim = [0, 0, 0]
        for i in range(len(self.ori_dim)):
            self.dim[i] = int(self.ori_dim[i] / self.spatial)

        self.num_workers = 16

        self.samples = [i for i in range(1, self.total_samples+1, self.temporal+1)]
        self.total_samples = self.samples[-1]
        self.num_samples_per_frame = (self.dim[0]*self.dim[1]*self.dim[2]//self.downsample_factor)//self.batch_size * self.batch_size
        
        self.queue_size = 2
        self.loader_queue = queue.Queue(maxsize=self.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=self.queue_size)

        self.data = self.preload_with_multi_threads(self.load_volume_data, num_workers=self.num_workers, data_str='Volume Data')
        self.data = torch.as_tensor(np.asarray(self.data), device=self.device, dtype=torch.float32)

        self.data_low = self._build_low_frequency_data(self.data, sigma=self.gaussian_sigma)
        self.data_high = self.data - self.data_low

        self.len = self.num_samples_per_frame * len(self.samples)
        self._get_data = self._get_training_data

        samples = self.ori_dim[2] * self.ori_dim[1] * self.ori_dim[0]
        self.coords = get_mgrid([self.ori_dim[0], self.ori_dim[1], self.ori_dim[2]], dim=3)
        self.time = np.zeros((samples, 1))
        self.testing_data_inputs = torch.as_tensor(np.concatenate((self.time, self.coords), axis=1),dtype=torch.float,device='cuda:0')
        self.preload_data()

    def _gaussian_kernel_1d(self, sigma, truncate=3.0):
        radius = int(truncate * sigma + 0.5)
        x = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def _build_low_frequency_data(self, data, sigma=1.0):
        if sigma <= 0:
            return data.clone()

        k1 = self._gaussian_kernel_1d(sigma)   # [K]
        kx = k1.view(1, 1, -1, 1, 1)
        ky = k1.view(1, 1, 1, -1, 1)
        kz = k1.view(1, 1, 1, 1, -1)

        radius = k1.numel() // 2
        x = data.unsqueeze(1)
        x = F.pad(x, (0, 0, 0, 0, radius, radius), mode='replicate')
        x = F.conv3d(x, kx)
        x = F.pad(x, (0, 0, radius, radius, 0, 0), mode='replicate')
        x = F.conv3d(x, ky)
        x = F.pad(x, (radius, radius, 0, 0, 0, 0), mode='replicate')
        x = F.conv3d(x, kz)
        return x.squeeze(1)

    @torch.no_grad()
    def _get_training_data(self):
        training_data_inputs = []
        training_data_outputs = []
        training_data_outputs_low = [] 
        training_data_outputs_high = [] 

        for i in range(0, len(self.samples)):
            x, y, z = fast_random_choice(self.dim, self.num_samples_per_frame)
            t = torch.ones_like(x) * (self.samples[i] - 1)

            outputs = self.data[i, x, y, z]

            outputs_low = self.data_low[i, x, y, z]
            outputs_high = self.data_high[i, x, y, z]

            x = x * self.spatial / (self.ori_dim[0] - 1)
            y = y * self.spatial / (self.ori_dim[1] - 1)
            z = z * self.spatial / (self.ori_dim[2] - 1)
            t = t / max((self.total_samples - 1), 1)

            inputs = torch.stack([t, x, y, z], dim=-1)
            inputs = 2.0 * inputs - 1.0

            training_data_inputs.append(inputs)
            training_data_outputs.append(outputs)

            training_data_outputs_low.append(outputs_low)
            training_data_outputs_high.append(outputs_high)

        training_data_inputs = torch.cat(training_data_inputs, dim=0).cuda()
        training_data_outputs = torch.cat(training_data_outputs, dim=0).cuda()
        training_data_outputs_low = torch.cat(training_data_outputs_low, dim=0).cuda()
        training_data_outputs_high = torch.cat(training_data_outputs_high, dim=0).cuda()
        idx = torch.randperm(training_data_inputs.shape[0], device='cpu')
        training_data_inputs = training_data_inputs[idx].contiguous()
        training_data_outputs = training_data_outputs[idx].contiguous()
        training_data_outputs_low = training_data_outputs_low[idx].contiguous()
        training_data_outputs_high = training_data_outputs_high[idx].contiguous()
        batchidxgenerator = BatchIndex(self.len, self.batch_size, shuffle=True)
        del idx
        cleanup()

        return training_data_inputs, {
            "final": training_data_outputs,
            "low": training_data_outputs_low,
            "high": training_data_outputs_high,
        }, batchidxgenerator

from torch import nn
import torch.optim as optim
import time
from torch.cuda.amp import autocast, GradScaler

def trainNet(model, args, dataset):
    result_dir = os.path.join(args.result_dir, f'{args.dataset}', f'OURS')   

    checkpoints_dir = os.path.join(result_dir, 'checkpoints')
    outputs_dir = os.path.join(result_dir, 'outputs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    loss_log_file = result_dir + '/' + 'loss.txt'
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6, fused=True)
    mse_loss = nn.MSELoss()
    scaler = GradScaler(enabled=args.fp16)
    
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        training_data_inputs, training_data_outputs, batchIndexGenerator = dataset.get_data()
        loss_mse = 0
        loop = tqdm.tqdm(batchIndexGenerator)

        for current_idx, next_idx in loop:
            coord = training_data_inputs[current_idx:next_idx].contiguous()

            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                v_final = training_data_outputs["final"][current_idx:next_idx].contiguous()
                v_low = training_data_outputs["low"][current_idx:next_idx].contiguous()
                v_high = training_data_outputs["high"][current_idx:next_idx].contiguous()

                pred_dict = model(coord)

                loss_final = mse_loss(pred_dict["final"].view(-1), v_final.view(-1))
                loss_low = mse_loss(pred_dict["low"].view(-1), v_low.view(-1))
                loss_high = mse_loss(pred_dict["high"].view(-1), v_high.view(-1))

                loss = args.lambda_final * loss_final + args.lambda_low * loss_low + args.lambda_high * loss_high
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_mse += loss.mean().item()
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_postfix(loss=loss_mse)

        adjust_lr(args, optimizer, epoch)

        with open(loss_log_file, "a") as f:
            f.write(f"Epochs {epoch}: loss = {loss_mse}, lr = {optimizer.param_groups[0]['lr']}")
            f.write('\n')

        if epoch % args.checkpoint == 0 or epoch == 1:
            torch.save(model.state_dict(),checkpoints_dir + '/' + '-' + str(args.spatial) + '-' + str(args.temporal) + '-' + str(epoch) + '.pth')
    elapsed = time.time() - start_time
    print(f"time: {elapsed:.4f}")

@torch.no_grad()
def inf(model, dataset, args, result_dir=None):
    ckpt = './FANR/result/' + args.dataset + args.ckpt + '-' + str(args.spatial) + '-' + str(args.temporal) + '-' + str(args.num_epochs) + '.pth'
    result_dir = os.path.dirname(os.path.dirname(ckpt)) if result_dir is None else result_dir
    outputs_dir = os.path.join(result_dir, 'outputs', str(args.spatial) + '-' + str(args.temporal))
    os.makedirs(outputs_dir, exist_ok=True)

    model.eval()
    samples = dataset.samples
    for i in range(len(samples)):
        for j in range(0, dataset.temporal + 1):
            frame_idx = samples[i] + j
            val_data_inputs, batchIndexGenerator = dataset._get_testing_data(frame_idx)
            d_final = []
            d_low = []     
            d_high = []   
            loop = tqdm.tqdm(batchIndexGenerator)
            for current_idx, next_idx in loop:
                coord = val_data_inputs[current_idx:next_idx]
                with torch.no_grad():
                    pred_dict = model(coord)
                    d_final.append(pred_dict["final"].view(-1))
                    d_low.append(pred_dict["low"].view(-1))
                    d_high.append(pred_dict["high"].view(-1))

            d_final = torch.cat(d_final, dim=-1).float().detach().cpu().numpy()
            d_final = np.asarray(d_final, dtype='<f')
            out_path = f'{outputs_dir}/{frame_idx:04}-OURS.raw'
            d_final.tofile(out_path, format='<f')

            d_low = torch.cat(d_low, dim=-1).float().detach().cpu().numpy()
            d_high = torch.cat(d_high, dim=-1).float().detach().cpu().numpy()
            # np.asarray(d_low, dtype='<f').tofile(f'{outputs_dir}/{frame_idx:04}-OURS-low.raw', format='<f')
            # np.asarray(d_high, dtype='<f').tofile(f'{outputs_dir}/{frame_idx:04}-OURS-high.raw', format='<f')