import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.network import DeformNet,AEDeformNet
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, load_pseudo_depth,get_bbox, compute_mAP, plot_mAP
import open3d as o3d
from lib.auto_encoder import PointCloudAE

def get_auto_encoder(model_path,emb_dim,n_pts):
    emb_dim = emb_dim
    n_pts = n_pts
    ae = PointCloudAE(emb_dim, n_pts)
    ae.cuda()
    ae.load_state_dict(torch.load(model_path))
    ae.eval()
    return ae

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=4096, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='/home/choisj/git/sj/object-deformnet/results/ae_train_4096_512/model_10.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=4096, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('assets/mean_points_emb.npy')

assert opt.data in ['val', 'real_test']
model_file_path = ['obj_models/real_test.pkl']



if opt.data == 'val':
    result_dir = 'results/eval_camera'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    # result_dir = 'results/eval_real'
    temp_dir_path = '_'.join(opt.model.split('/')[-2:])[:-4]
    result_dir = 'results/eval_' + temp_dir_path
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def detect():
    models = {}
    for path in model_file_path:
        with open(os.path.join(opt.data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    models = models
    # resume model
    viz_pcd = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    opt.emb = 512
    estimator = AEDeformNet(opt.n_cat, opt.nv_prior, opt.emb)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    # frame by frame test
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_pseudo_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))        
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_shape = np.zeros((num_insts, 2048, 3), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
        # prepare frame data
        f_points, f_rgb, f_choose, f_catId, f_prior,f_model = [], [], [], [], [], []
        valid_inst = []
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            prior = mean_shapes[cat_id]
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            model = models[gts['model_list'][i]].astype(np.float32)     # 1024 points
            
            # no depth observation for background in CAMERA dataset
            # beacuase of how we compute the bbox in function get_bbox
            # there might be a chance that no foreground points after cropping the mask
            # cuased by false positive of mask_rcnn, most of the regions are background
            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            # process objects with valid depth observation
            if len(choose) > opt.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')
            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            points = np.concatenate((pt0, pt1, pt2), axis=1)
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)
            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)
            # concatenate instances
            f_points.append(points)
            f_rgb.append(rgb)
            f_choose.append(choose)
            f_catId.append(cat_id)
            f_prior.append(prior)
            f_model.append(model)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            f_catId = torch.cuda.LongTensor(f_catId)
            f_prior = torch.cuda.FloatTensor(f_prior)
            f_model = torch.cuda.FloatTensor(f_model)
            # inference
            torch.cuda.synchronize()
            t_now = time.time()
            
            embedding, point_cloud = estimator(f_points, f_model)
            # assign_mat, deltas = estimator(f_points, f_rgb, f_choose, f_catId, f_prior)
            
            pcd = o3d.geometry.PointCloud()
            for i in range(num_insts):
                pcd.points = o3d.utility.Vector3dVector(f_points[3].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
                pcd.points = o3d.utility.Vector3dVector(f_model[3].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
                pcd.points = o3d.utility.Vector3dVector(point_cloud[3].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
            
            

    # write statistics
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference/img_count))
    messages.append("Umeyama time: {:06f}  Average: {:06f}/image".format(t_umeyama, t_umeyama/img_count))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_10_idx] * 100))
    
    messages.append('1')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[1, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[1, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[1, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[1, degree_10_idx, shift_10_idx] * 100))

    messages.append('2')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[2, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[2, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[2, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[2, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[2, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[2, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[2, degree_10_idx, shift_10_idx] * 100))


    messages.append('3')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[3, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[3, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[3, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[3, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[3, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[3, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[3, degree_10_idx, shift_10_idx] * 100))

    messages.append('4')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[4, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[4, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[4, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[4, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[4, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[4, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[4, degree_10_idx, shift_10_idx] * 100))

    messages.append('5')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[5, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[5, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[5, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[5, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[5, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[5, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[5, degree_10_idx, shift_10_idx] * 100))
    
    messages.append('6')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[6, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[6, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[6, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[6, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[6, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[6, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[6, degree_10_idx, shift_10_idx] * 100))
    
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    # load NOCS results
    pkl_path = os.path.join('results/nocs_results', opt.data, 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)
    nocs_iou_aps = nocs_results['iou_aps'][-1, :]
    nocs_pose_aps = nocs_results['pose_aps'][-1, :, :]
    iou_aps = np.concatenate((iou_aps, nocs_iou_aps[None, :]), axis=0)
    pose_aps = np.concatenate((pose_aps, nocs_pose_aps[None, :, :]), axis=0)
    # plot
    plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':
    print('Detecting ...')
    print(torch.cuda.is_available())
    detect()
    print('Evaluating ...')
    evaluate()