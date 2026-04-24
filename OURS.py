from OURS_base.train import ScalarDataSet, trainNet, inf
from OURS_base.model import FINR,FINR_SS
from utils import seed_everything

from pathlib import Path
import numpy as np
import torch

DATASET_CONFIG = {
    "fivejets":   (128, 128, 128),
    "vortex":     (128, 128, 128),
    "tornado":    (128, 128, 128),
    "tangaroa":   (300, 180, 120),
    "h2":         (600, 248, 248),
    "T":          (600, 248, 248),
    "bubble":     (640, 256, 256),
    "supernova":  (204, 204, 204),
    "combustion": (480, 720, 120),
    "yoh":        (480, 720, 120),
    "halfcy":     (640, 240, 80),
    "magnetic":   (512, 512, 512),
}


def evaluate(dataset_name, spa, tem):
    root_dir = Path(__file__).resolve().parent

    gt_path = root_dir / "dataset" / dataset_name / "0001.raw"
    pred_dir = root_dir / "result" / dataset_name / "OURS" / "outputs" / f"{spa}-{tem}"
    pred_path = pred_dir / "0001-OURS.raw"
    save_path = pred_dir / "OURS.raw"

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X, Y, Z = DATASET_CONFIG[dataset_name]

    gt = np.fromfile(gt_path, dtype=np.float32)
    pred = np.fromfile(pred_path, dtype=np.float32)

    gt = 2 * (gt - np.min(gt)) / (np.max(gt) - np.min(gt)) - 1

    gt_t = torch.from_numpy(gt)
    pred_t = torch.from_numpy(pred)

    diff = gt_t.max() - gt_t.min()
    psnr = 10.0 * torch.log10(diff**2 / torch.mean((gt_t - pred_t) ** 2))
    psnr_value = round(psnr.item(), 2)

    print(f"PSNR of {dataset_name}: {psnr_value}")

    pred_save = pred.copy()
    pred_save[0] = -1.0
    pred_save[1] = 1.0
    pred_save.astype(np.float32).tofile(save_path)

    return psnr.item()

import argparse
p = argparse.ArgumentParser()
p.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
p.add_argument('--gpu', type=str,default='0')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--fp16', action="store_true")
p.add_argument('--batch_size', type=int, default=8000)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for.')
p.add_argument('--checkpoint', type=int, default=50, help='checkpoint is saved.')
p.add_argument('--ckpt', type=str,default="/OURS/checkpoints/",help='checkpoint path.')
p.add_argument('--result_dir', type=str, default='./FINR/result/', metavar='N',
                    help='the path where we stored the synthesized data')
p.add_argument('--temporal', type=int, default=0, metavar='N')
p.add_argument('--lr_s', type=str, default='cosine', help='learning rate scheduler')
p.add_argument('--gaussian_sigma', type=float, default=15, help='sigma for Gaussian smoothing')
p.add_argument('--lambda_final', type=float, default=1, help='loss weight for final reconstruction')
p.add_argument('--lambda_low', type=float, default=1, help='loss weight for low-frequency prediction')
p.add_argument('--lambda_high', type=float, default=1, help='loss weight for high-frequency prediction')

p.add_argument('--dataset', type=str, default='vortex')
p.add_argument('--spatial', type=int, default=1, metavar='N')
p.add_argument('--mode', type=str, default='train', metavar='N')
opt = p.parse_known_args()[0]

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
seed_everything(opt.seed)
torch.set_float32_matmul_precision('high')

def main():
    Data = ScalarDataSet(opt)
    Model = FINR(4,1,init_features=208, num_res=1)
    # Model = FINR_SS(4,1,init_features=148, num_res=1)
    Model.cuda()

    if opt.mode == 'train':
        print('Initalize Model Successfully.')
        trainNet(Model,opt,Data)
        inf(Model, Data,opt)
        evaluate(opt.dataset, opt.spatial, opt.temporal)

    elif opt.mode == 'eval':
        evaluate(opt.dataset, opt.spatial, opt.temporal)

if __name__== "__main__":
    main()