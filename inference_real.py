import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
# import pytorch_lightning as pl
import _pickle as cPickle
import sys
# sys.path.append("simnet")
# from simnet.lib.net import common
# from simnet.lib import camera
# from simnet.lib.net.panoptic_trainer import PanopticModel
# from simnet.lib.net.models.auto_encoder import PointCloudAE
# from utils.nocs_utils import load_img_NOCS, create_input_norm
# from utils.viz_utils import depth2inv, viz_inv_depth
# from utils.transform_utils import get_gt_pointclouds, transform_coordinates_3d, calculate_2d_projections
# from utils.transform_utils import project
# from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh, draw_gt_bboxes
# from utils.nocs_eval_utils import draw_detections
import time

import torchvision.transforms as transforms
from lib.auto_encoder import PointCloudAE
from lib.network import DeformNet
from utils import camera
from lib.utils import draw_detections,align_rotation, transform_coordinates_3d,calculate_2d_projections,get_3d_bbox
from utils.viz_utils import save_projected_points,line_set_mesh,draw_bboxes,draw_axes,draw_bboxes_origin
from utils.transform_utils import get_gt_pointclouds,project

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CAMERA+Real', help='CAMERA or CAMERA+Real')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=25, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--checkpoint', type=str, default='/home/choisj/git/sj/object-deformnet/results/real/model_50.pth', help='evaluate model')
parser.add_argument('--result_dir', type=str, default='results/camera_real', help='directory to save train results')

mean_shapes = np.load('assets/mean_points_emb.npy')
opt = parser.parse_args()
assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    result_dir = 'results/eval_camera'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = 'results/eval_real'
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

def get_auto_encoder(model_path):
  emb_dim = 512
  n_pts = 1024
  ae = PointCloudAE(emb_dim, n_pts)
  ae.cuda()
  ae.load_state_dict(torch.load(model_path))
  ae.eval()
  return ae

def inference(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
    model = DeformNet(opt.n_cat, opt.nv_prior)
    model.eval()
    
    if use_gpu:
        model.cuda()
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    
    _CAMERA = camera.NOCS_Real()
    for i, img_path in enumerate(data_path):
        
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png' 
        if not os.path.exists(color_path):
            continue
        img_vis = cv2.imread(color_path)
        
        img_path_parsing = img_full_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
            
        image_short_path = '_'.join(img_path_parsing[-3:])
        load_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(load_path, 'rb') as f:
            result = cPickle.load(f)
        cv2.imwrite(
            str(output_path / f'{i}_image.png'),
            np.copy(np.copy(img_vis))
        )
        
        write_pcd = False
        rotated_pcds = []
        points_2d = []
        box_obb = []
        axes = []
        gt_axes_li = []
        
        num_insts = len(mrcnn_result['class_ids'])
            
        for j in range(num_insts):
            shape_out = result['pred_shape'][j]
            
            rotated_pc, rotated_box, pred_size = get_gt_pointclouds(result['pred_RTs'][j],shape_out,camera_model=_CAMERA)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(rotated_pc)
            filename_rotated = str(output_path) + '/pcd_rotated'+str(i)+str(j)+'.ply'
            if write_pcd:
                o3d.io.write_point_cloud(filename_rotated, pcd)
            else:
                rotated_pcds.append(pcd)

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            T = result['pred_RTs'][j]
            mesh_frame = mesh_frame.transform(T)
            rotated_pcds.append(mesh_frame)
            cylinder_segments = line_set_mesh(rotated_box)
            for k in range(len(cylinder_segments)):
                rotated_pcds.append(cylinder_segments[k])

                
            points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
            points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
            points_2d_mesh = points_2d_mesh.T
            points_2d.append(points_2d_mesh)
            #2D output
            points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
            points_2d_obb = project(_CAMERA.K_matrix, points_obb)
            points_2d_obb = points_2d_obb.T
            box_obb.append(points_2d_obb)
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            sRT = result['pred_RTs'][j]
            transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
            projected_axes = calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3])

            
            axes.append(projected_axes)
            #RT output
        for k in range(result['gt_RTs'].shape[0]):
            gt_axes = transform_coordinates_3d(xyz_axis,result['gt_RTs'][k])
            gt_projected_axes = calculate_2d_projections(gt_axes,_CAMERA.K_matrix[:3,:3])
            gt_axes_li.append(gt_projected_axes)
            
        if not write_pcd:
        # o3d.visualization.draw_geometries(rotated_pcds)
            save_projected_points(np.copy(img_vis), points_2d, str(output_path), i, result['pred_class_ids'])
        
            colors_box = [(0,0,220)]
            im = np.array(np.copy(img_vis)).copy()
            
            # Draw Pred BBoxes, Axes
            for k in range(len(colors_box)):
                for points_2d, axis in zip(box_obb, axes):
                    points_2d = np.array(points_2d)
                    im = draw_bboxes(im, points_2d, axis, colors_box[k])
                    
            # Draw GT Axes
            for k in range(len(colors_box)):
                for points_2d, axis in zip(result['gt_bboxes'], gt_axes_li):
                    points_2d = np.array(points_2d)
                    im = draw_axes(im, points_2d, axis, colors_box[k])
                    
            # Draw GT BBoxes 
            for k in range(result['gt_RTs'].shape[0]):
                if result['gt_class_ids'][k] in [1, 2, 4]:
                    sRT = align_rotation(result['gt_RTs'][k, :, :])
                else:
                    sRT = result['gt_RTs'][k, :, :]
                bbox_3d = get_3d_bbox(result['gt_scales'][k, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, _CAMERA.K_matrix[:3,:3])
                im = draw_bboxes_origin(im, projected_bbox, (0, 255, 0))
            box_plot_name = str(output_path)+'/box3d'+str(i).zfill(3)+'.png'
            cv2.imwrite(
                box_plot_name,
                np.copy(im)
            )
        print("done with image: ", i )

if __name__ == '__main__':
  print(opt)
  result_name = 'inference'
  path = 'data/'+result_name
  output_path = pathlib.Path(path) / opt.checkpoint[-7:-5]
  output_path.mkdir(parents=True, exist_ok=True)
  inference(opt, opt.data_dir, output_path)
