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
from concurrent.futures import ThreadPoolExecutor

class ScalarDataSet(LoadData):
    def __init__(self,args, device='cuda:0'):
        self.dataset, self.batch_size = args.dataset, args.batch_size
        self.temporal, self.spatial = args.temporal, args.spatial
        self.device = device        
        self.ori_dim, self.total_samples, self.data_path, self.downsample_factor = dataset_selection(self.dataset,self.spatial, 
                                                                                                     self.temporal)
        self.dim = [0,0,0]    
        for i in range(len(self.ori_dim)):
            self.dim[i] = int(self.ori_dim[i] / self.spatial)

        self.num_workers = 16

        self.samples = [i for i in range(1,self.total_samples+1,self.temporal+1)]
        self.total_samples = self.samples[-1]
        self.num_samples_per_frame = (self.dim[0]*self.dim[1]*self.dim[2]//self.downsample_factor)//self.batch_size * self.batch_size

        self.queue_size = 2
        self.loader_queue = queue.Queue(maxsize=self.queue_size) 
        self.executor = ThreadPoolExecutor(max_workers=self.queue_size)

        self.data = self.preload_with_multi_threads(self.load_volume_data, num_workers=self.num_workers, data_str='Volume Data')
        self.data = torch.as_tensor(np.asarray(self.data), device=self.device)  

        self.len = self.num_samples_per_frame * len(self.samples)
        self._get_data = self._get_training_data

        samples = self.ori_dim[2]*self.ori_dim[1]*self.ori_dim[0]
        self.coords = get_mgrid([self.ori_dim[0],self.ori_dim[1],self.ori_dim[2]],dim=3)
        self.time = np.zeros((samples,1))
        self.testing_data_inputs = torch.as_tensor(np.concatenate((self.time, self.coords),axis=1), dtype=torch.float, device='cuda:0')
        self.preload_data()
        
    @torch.no_grad()
    def _get_training_data(self):
        training_data_inputs = []
        training_data_outputs = []

        for i in range(0, len(self.samples)):
            x,y,z = fast_random_choice(self.dim, self.num_samples_per_frame)
            t = torch.ones_like(x) * (self.samples[i]-1)

            outputs = self.data[i, x, y, z]  

            x = x * self.spatial / (self.ori_dim[0] - 1) 
            y = y * self.spatial / (self.ori_dim[1] - 1)  
            z = z * self.spatial / (self.ori_dim[2] - 1)  
            t = t / max((self.total_samples-1), 1)

            inputs = torch.stack([t, x, y, z], dim=-1)
            inputs = 2.0 * inputs - 1.0  
            training_data_inputs.append(inputs)
            training_data_outputs.append(outputs)

        training_data_inputs = torch.cat(training_data_inputs, dim=0).cuda()
        training_data_outputs = torch.cat(training_data_outputs, dim=0).cuda()
        idx = torch.randperm(training_data_inputs.shape[0], device='cpu')
        training_data_inputs = training_data_inputs[idx].contiguous()
        training_data_outputs = training_data_outputs[idx].contiguous()
        batchidxgenerator = BatchIndex(self.len, self.batch_size, shuffle=True)
        del idx
        cleanup()
        return training_data_inputs, training_data_outputs, batchidxgenerator

import torch
from torch import nn
import os
import numpy as np
import torch.optim as optim
import tqdm
import time
from torch.cuda.amp import autocast, GradScaler

def trainNet(model,args,dataset):
    result_dir = os.path.join(args.result_dir, f'{args.dataset}', f'{args.model}')

    checkpoints_dir = os.path.join(result_dir, 'checkpoints')
    outputs_dir = os.path.join(result_dir, 'outputs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    loss_log_file = result_dir+'/'+'loss.txt'
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-6, fused=True)
    mse_loss = nn.MSELoss()
    scaler = GradScaler(enabled=args.fp16)
    
    start_time = time.time()
    for epoch in range(1,args.num_epochs+1):
        model.train()
        training_data_inputs, training_data_outputs, batchIndexGenerator = dataset.get_data()
        loss_mse = 0
        loop = tqdm.tqdm(batchIndexGenerator)

        for current_idx, next_idx in loop:
            coord = training_data_inputs[current_idx:next_idx].contiguous()
            v = training_data_outputs[current_idx:next_idx].contiguous()
            
            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                v_pred = model(coord)
                loss = mse_loss(v_pred.view(-1),v.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_mse += loss.mean().item()

            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_postfix(loss=loss_mse)
        adjust_lr(args, optimizer, epoch)

        with open(loss_log_file,"a") as f:
            f.write(f"Epochs {epoch}: loss = {loss_mse}, lr = {optimizer.param_groups[0]['lr']}")
            f.write('\n')

        if epoch%args.checkpoint == 0 or epoch == 1:
            torch.save(model.state_dict(),checkpoints_dir+'/'+'-'+str(args.spatial)+'-'+str(args.temporal)+'-'+str(epoch)+'.pth')
    elapsed = time.time() - start_time
    print(f"time: {elapsed:.4f}")

@torch.no_grad()
def inf(model,dataset,args, result_dir=None):
    ckpt = './FINR/result/'+args.dataset+'/'+args.model+'/checkpoints/-'+str(args.spatial)+'-'+str(args.temporal)+'-'+str(args.num_epochs)+'.pth'
    result_dir = os.path.dirname(os.path.dirname(ckpt)) if result_dir is None else result_dir
    outputs_dir = os.path.join(result_dir, 'outputs', str(args.spatial)+'-'+str(args.temporal))
    os.makedirs(outputs_dir, exist_ok=True)

    model.eval()
    samples = dataset.samples
    for i in range(len(samples)):  
        for j in range(0,dataset.temporal+1):
            frame_idx = samples[i] + j
            val_data_inputs, batchIndexGenerator =dataset._get_testing_data(frame_idx)
            d = []
            loop = tqdm.tqdm(batchIndexGenerator)
            for current_idx, next_idx in loop:
                coord = val_data_inputs[current_idx:next_idx]
                with torch.no_grad():
                    dat = model(coord).view(-1)
                    d.append(dat)
            d = torch.cat(d,dim=-1).float()
            d = d.detach().cpu().numpy()
            d = np.asarray(d,dtype='<f')
            out_path = f'{outputs_dir}/{frame_idx:04}-{args.model}.raw'
            d.tofile(out_path, format='<f')