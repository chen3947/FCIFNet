import torch
import torch.nn.functional as F
import sys
import warnings
from tqdm import tqdm
import numpy as np
import os
import argparse
import cv2
import time
from model.FCIFNet import FCIFNet
from data import test_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--test_path', type=str, default='test_data', help='test dataset path')
parser.add_argument('--save_path', type=str, default='./test_maps/FCIFNet-RGBD/', help='save path')
parser.add_argument('--pth_path', type=str, default=None, help='checkpoint path')
opt = parser.parse_args()

dataset_path = opt.test_path

if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

model = FCIFNet().eval().cuda()
state_dict = torch.load(opt.pth_path)

filtered_state_dict = {k: v for k, v in state_dict.items()
                      if not k.endswith('total_ops') and not k.endswith('total_params')}
model.load_state_dict(filtered_state_dict, strict=True)

test_datasets = ['DUT', 'LFSD', 'NJU2K', 'NLPR', 'SIP', 'STERE']

for dataset in test_datasets:
    save_path = os.path.join(opt.save_path, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset, 'RGB/')
    gt_root = os.path.join(dataset_path, dataset, 'GT/')
    depth_root = os.path.join(dataset_path, dataset, 'depth/')

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.trainsize)

    dummy_image = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
    dummy_depth = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
    for _ in range(10):
        _ = model(dummy_image, dummy_depth)
    torch.cuda.synchronize()

    total_time = 0
    total_frames = 0

    for i in tqdm(range(test_loader.size), desc=dataset, file=sys.stdout):
        image, gt, depth, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()

        torch.cuda.synchronize() 
        start_time = time.time()

        with torch.no_grad():
            res, pred_2, pred_3, pred_4 = model(image, depth)

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_frames += 1

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(save_path, name), res * 255)

    avg_fps = total_frames / total_time
    print(f"{dataset} Average Inference FPS: {avg_fps:.2f}")

    print(f'Test Done for {dataset}!')
