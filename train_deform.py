import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from lib.network import DeformNet
from lib.loss import Loss
from data.pose_dataset import PoseDataset
from lib.utils import setup_logger, compute_sRT_errors
from lib.align import estimateSimilarityTransform
import configparser
import json
from tqdm import tqdm
import wandb

class ArgsNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def train_net(config):
    opt = configparser.ConfigParser()
    opt.read(config.config)
    opt = opt._sections['Arguments']
    opt = ArgsNamespace(**opt)
    opt.n_pts = int(opt.n_pts)
    opt.n_cat = int(opt.n_cat)
    opt.nv_prior= int(opt.nv_prior)
    opt.img_size=int(opt.img_size)
    opt.batch_size=int(opt.batch_size)
    opt.num_workers=int(opt.num_workers)
    opt.start_epoch=int(opt.start_epoch)
    opt.max_epoch=int(opt.max_epoch)
    opt.lr = float(opt.lr)
    opt.decay_rate = json.loads(opt.decay_rate)
    opt.decay_epoch = json.loads(opt.decay_epoch)
        
    if opt.wandb =='online':
        wandb.init(project='phocl-deform') 
        wandb.run.name = 'gt_model'
        
    # opt.decay_epoch = [0,3, 5]
    # opt.decay_epoch = [0,5,10,15,20]
    # opt.decay_rate = [1.0, 0.6, 0.01]
    # opt.decay_rate = [1.0, 0.6, 0.3, 0.1, 0.01]
    opt.corr_wt = 1.0
    opt.cd_wt = 5.0
    opt.entropy_wt = 0.0001
    opt.deform_wt = 1.0
    # opt.deform_wt = 0.01
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    estimator = DeformNet(opt.n_cat, opt.nv_prior)
    estimator.cuda()
    criterion = Loss(opt.corr_wt, opt.cd_wt, opt.entropy_wt, opt.deform_wt)
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))
    # dataset
    train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size)
    # train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size)
    val_dataset = PoseDataset(opt.dataset, 'test', opt.data_dir, opt.n_pts, opt.img_size)
    # val_dataset = PoseDataset('CAMERA+Real', 'test', opt.data_dir, opt.n_pts, opt.img_size)
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
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate if needed
        if decay_count < len(opt.decay_rate):
            if epoch > opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if opt.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len+real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3*n_repeat*real_len) + real_indices*n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start+train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                                                       num_workers=opt.num_workers, pin_memory=True)
        estimator.train()
        # for i, data in tqdm(enumerate(train_dataloader, 1)):
        with tqdm(train_dataloader, unit='batch') as tepoch:
            for i,data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                points, rgb, choose, cat_id, model, prior, sRT, nocs = data
                points = points.cuda()
                rgb = rgb.cuda()
                choose = choose.cuda()
                cat_id = cat_id.cuda()
                model = model.cuda()
                prior = prior.cuda()
                sRT = sRT.cuda()
                nocs = nocs.cuda()
                # assign_mat, deltas = estimator(points, rgb, choose, cat_id, prior)
                assign_mat, deltas = estimator(points, rgb, choose, cat_id, model)
                loss, corr_loss, cd_loss, entropy_loss, deform_loss = criterion(assign_mat, deltas, prior, nocs, model)
                optimizer.zero_grad()
                if opt.wandb == 'online':
                    wandb.log({
                        'learning_rate' : current_lr,
                        'train_loss' : loss,
                        'corr_loss' : corr_loss,
                        'cd_loss' : cd_loss,
                        'entropy_loss' : entropy_loss,
                        'deform_loss' : deform_loss
                    })
                loss.backward()
                optimizer.step()
                global_step += 1
                tepoch.set_postfix(loss=loss.item(),corr_loss=corr_loss.item(), cd_loss=cd_loss.item(), entropy_loss=entropy_loss.item(), deform_loss=deform_loss.item())

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))
        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) +
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))
        val_loss = 0.0
        total_count = np.zeros((opt.n_cat,), dtype=int)
        strict_success = np.zeros((opt.n_cat,), dtype=int)           # 5 degree and 5 cm
        strict_easy_success = np.zeros((opt.n_cat,), dtype=int)      # 5 degree and 10 cm
        easy_success = np.zeros((opt.n_cat,), dtype=int)             # 10 degree and 5 cm
        easy_easy_success = np.zeros((opt.n_cat,), dtype=int)        # 10 degree and 10 cm
        iou_success = np.zeros((opt.n_cat,), dtype=int)              # relative scale error < 0.1
        # sample validation subset
        val_size = 1500
        val_idx = random.sample(list(range(val_dataset.length)), val_size)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                                                     num_workers=opt.num_workers, pin_memory=True)
        estimator.eval()
        with tqdm(val_dataloader, unit='batch') as tepoch:
            for i,data in enumerate(tepoch):
                points, rgb, choose, cat_id, model, prior, sRT, nocs = data
                points = points.cuda()
                rgb = rgb.cuda()
                choose = choose.cuda()
                cat_id = cat_id.cuda()
                model = model.cuda()
                prior = prior.cuda()
                sRT = sRT.cuda()
                nocs = nocs.cuda()
                assign_mat, deltas = estimator(points, rgb, choose, cat_id, model)
                # assign_mat, deltas = estimator(points, rgb, choose, cat_id, prior)
                loss, _, _, _, _ = criterion(assign_mat, deltas, prior, nocs, model)
                # estimate pose and scale
                inst_shape = prior + deltas
                assign_mat = F.softmax(assign_mat, dim=2)
                nocs_coords = torch.bmm(assign_mat, inst_shape)
                nocs_coords = nocs_coords.detach().cpu().numpy()[0]
                points = points.cpu().numpy()[0]
                # use choose to remove repeated points
                choose = choose.cpu().numpy()[0]
                _, choose = np.unique(choose, return_index=True)
                nocs_coords = nocs_coords[choose, :]
                points = points[choose, :]
                _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, points)
                # evaluate pose
                cat_id = cat_id.item()
                if pred_sRT is not None:
                    sRT = sRT.detach().cpu().numpy()[0]
                    R_error, T_error, IoU = compute_sRT_errors(pred_sRT, sRT)
                    if R_error < 5 and T_error < 0.05:
                        strict_success[cat_id] += 1
                    if R_error < 5 and T_error < 0.1:
                        strict_easy_success[cat_id] += 1
                    if R_error < 10 and T_error < 0.1:
                        easy_easy_success[cat_id] += 1
                    if R_error < 10 and T_error < 0.05:
                        easy_success[cat_id] += 1
                    if IoU < 0.1:
                        iou_success[cat_id] += 1
                total_count[cat_id] += 1
                val_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        # compute accuracy
        strict_acc = 100 * (strict_success / total_count)
        strict_easy_acc = 100 * (strict_easy_success / total_count)
        easy_acc = 100 * (easy_success / total_count)
        easy_easy_acc = 100 * (easy_easy_success / total_count)
        iou_acc = 100 * (iou_success / total_count)
        for i in range(opt.n_cat):
            logger.info('{} accuracies:'.format(val_dataset.cat_names[i]))
            logger.info('5^o 5cm: {:4f}'.format(strict_acc[i]))
            logger.info('5^o 10cm: {:4f}'.format(strict_easy_acc[i]))
            logger.info('10^o 5cm: {:4f}'.format(easy_acc[i]))
            logger.info('10^o 10cm: {:4f}'.format(easy_easy_acc[i]))
            logger.info('IoU < 0.1: {:4f}'.format(iou_acc[i]))
        strict_acc = np.mean(strict_acc)
        strict_easy_acc = np.mean(strict_easy_acc)
        easy_acc = np.mean(easy_acc)
        easy_easy_acc = np.mean(easy_easy_acc)
        iou_acc = np.mean(iou_acc)
        val_loss = val_loss / val_size
        if opt.wandb == 'online':
            wandb.log({
                    'val_loss' : current_lr,
                    '5^o5cm_acc' : strict_acc,
                    '5^o10cm_acc' : strict_easy_acc,
                    '10^o5cm_acc' : easy_acc,
                    '10^o10cm_acc' : easy_easy_acc,
                    'iou_acc' : iou_acc,
                })
        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('Overall accuracies:')
        logger.info('5^o 5cm: {:4f} 5^o 10cm: {:4f} 10^o 5cm: {:4f} 10^o 10cm: {:4f} IoU: {:4f}'.format(strict_acc, strict_easy_acc, easy_acc, easy_easy_acc, iou_acc))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default= 'configs/phocal_2048.ini', type=str, help='path to configuration file')
    opt = parser.parse_args()
    train_net(opt)
