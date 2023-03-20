import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from lib.network import DeformNet,AEDeformNet
from lib.loss import Loss
from data.pose_dataset import PoseDataset
from lib.utils import setup_logger, compute_sRT_errors
from lib.align import estimateSimilarityTransform
from lib.auto_encoder import PointCloudAE
from tqdm import tqdm
import wandb
from lib.loss import ChamferLoss
import open3d as o3d
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Real', help='CAMERA or CAMERA+Real')
# parser.add_argument('--dataset', type=str, default='Real', help='CAMERA or CAMERA+Real')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_pts', type=int, default=4096, help='number of foreground points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=4096, help='number of vertices in shape priors')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
# parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
parser.add_argument('--max_epoch', type=int, default=12, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='/home/choisj/git/sj/object-deformnet/results/ae_train_4096_512/model_12.pth', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/ae_train_4096_512', help='directory to save train results')
parser.add_argument('--wandb', type=str, default='offline', help='wandb online mode')
opt = parser.parse_args()

opt.decay_epoch = [0, 6, 10]
opt.decay_rate = [1.0, 0.6, 0.1]
opt.corr_wt = 1.0
opt.cd_wt = 5.0
opt.entropy_wt = 0.0001
opt.deform_wt = 0.01
opt.emb = 512

if opt.wandb =='online':
    wandb.init(project='object-deform') 
    wandb.run.name = 'finetune fusion depth + pseudo depth'
    
def train_net():
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    estimator = AEDeformNet(opt.n_cat, opt.nv_prior, opt.emb)
    estimator.cuda()
    criterion = ChamferLoss()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))
    # dataset
    val_dataset = PoseDataset('Real', 'test', opt.data_dir, opt.n_pts, opt.img_size)
    # val_dataset = PoseDataset(opt.dataset, 'test', opt.data_dir, opt.n_pts, opt.img_size)
    # start training
    st_time = time.time()
    train_steps = 1500
    global_step = train_steps * (opt.start_epoch - 1)
    n_decays = len(opt.decay_epoch)
    assert len(opt.decay_rate) == n_decays
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i
    train_size = train_steps * opt.batch_size
    indices = []
    page_start = -train_size
    # dataset
    val_size = 340
    val_idx = random.sample(list(range(val_dataset.length)), val_size)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                                                    num_workers=opt.num_workers, pin_memory=True)
    
    pcd = o3d.geometry.PointCloud()
    estimator.eval()
    with torch.no_grad():
        with tqdm(val_dataloader, unit='batch') as tepoch:
            for i,data in enumerate(tepoch):
                points, rgb, choose, cat_id, model, prior, sRT, nocs = data
                points = points.cuda()
                model = model.cuda()
                
                embedding, point_cloud = estimator(points, model)
                loss,_,_ = criterion(point_cloud, model)
                
                pcd.points = o3d.utility.Vector3dVector(points[0].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
                pcd.points = o3d.utility.Vector3dVector(model[0].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
                pcd.points = o3d.utility.Vector3dVector(point_cloud[0].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
        
        

if __name__ == '__main__':
    train_net()
