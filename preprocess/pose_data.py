import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
sys.path.append('../lib')
from align import align_nocs_to_depth
from utils import load_depth,calculate_xyz_bbox_size
import json

train_obj = {'box_leibniz_choco', 'box_principe', 'can_couscous', 'bottle_eres_inox', 'can_meatball', 'box_crunchy_muesli', 'cup_green_actys', 'box_koala', 'cutlery_spoon_2', 'glass_jar_big', 'remote_tv_white_quelle', 'teapot_wooden_color', 'cutlery_fork_3', 'cup_stanford', 'remote_grey', 'cutlery_fork_2', 'box_barilla', 'cup_white_whisker', 'glass_green_bottle', 'glass_beer_mug', 'cup_yellow_white_border', 'cutlery_knife_2', 'remote_aircon_chunghop', 'glass_ashtray_big', 'can_fanta', 'bottle_evian_frozen', 'cutlery_spoon_1', 'cup_yellow_handle', 'can_large_tuna', 'teapot_white_blue_cone_top', 'cutlery_knife_1'}
test_obj = {'remote_jaxster', 'can_monster', 'bottle_evian_red', 'teapot_big_white_floral', 'remote_toy', 'can_sheba_cat', 'remote_infini_fun', 'bottle_85_alcool', 'teapot_white_was_brand', 'teapot_brown_chinese', 'cup_green_actys', 'cutlery_spatula', 'bottle_soupline', 'can_tuna_salad', 'cutlery_scissors', 'glass_small_5', 'cup_new_york_big', 'cutlery_fork_3', 'box_proschmelz', 'can_corned_beef', 'bottle_dettol_washing_machine', 'glass_jar_landliebe', 'glass_green_bottle', 'cutlery_knife_2', 'cutlery_fork_1', 'cutlery_peeler', 'glass_cup_small_2', 'cup_plastic_green_flowers', 'box_special_k', 'cup_red_heart', 'box_antikalk', 'glass_cocktail', 'teapot_ambition_brand'}

root_dir = '/home/choisj/git/sj/phocal_object_deformnet/object-deformnet/data/'
train_sequence_idx = [1,2,3,4,5,6,7,8,9,10,11,12]
test_sequence_idx = [13,14,15,16,17,18,19,20,21,22,23,24]
train_intrinsic = []
test_intrinsic = []

for idx in train_sequence_idx:
    intrinsic_path = os.path.join(root_dir,'train','sequence_'+str(idx),'origin','scene_camera.json')
    with open(intrinsic_path,'r') as f:
        intrinsic_file = json.load(f)
    fx = intrinsic_file['rgb']['fx']
    fy = intrinsic_file['rgb']['fy']
    cx = intrinsic_file['rgb']['cx']
    cy = intrinsic_file['rgb']['cy']
    
    tmp_intrinsic = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    train_intrinsic.append(tmp_intrinsic)
    f.close()

for idx in test_sequence_idx:
    intrinsic_path = os.path.join(root_dir,'test','sequence_'+str(idx),'origin','scene_camera.json')
    with open(intrinsic_path,'r') as f:
        intrinsic_file = json.load(f)
    fx = intrinsic_file['rgb']['fx']
    fy = intrinsic_file['rgb']['fy']
    cx = intrinsic_file['rgb']['cx']
    cy = intrinsic_file['rgb']['cy']
    
    tmp_intrinsic = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    test_intrinsic.append(tmp_intrinsic)
    f.close()

train_intrinsic = np.array(train_intrinsic)
test_intrinsic = np.array(test_intrinsic)

def create_img_list(data_dir):
    """ Create train/val/test data list for PhoCaL. """
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir,subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[0] == 0
    del all_inst_ids[0]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754' or model_id == 'd3b53f56b4a7b3b3c9f016d57db96408':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_phocal_train(data_dir):
    """ Generate gt labels for Real train data through PnP. """
    phocal_train = open(os.path.join(data_dir, 'train_list_all.txt')).read().splitlines()
    # scale factors for all instances
    scale_factors = {}
    obj_list = os.listdir(os.path.join(data_dir,'obj_models_small_size'))
    for obj in obj_list:
        path_to_size = glob.glob(os.path.join(data_dir, 'obj_models_small_size',obj, '*.obj'))
        for inst_path in sorted(path_to_size):
            instance = os.path.basename(inst_path).split('.')[0]
            if instance in train_obj:
                x,y,z = calculate_xyz_bbox_size(inst_path)
                bbox_dims = np.array((x,y,z))
                scale_factors[instance] = np.linalg.norm(bbox_dims)

    valid_img_list = []
    for img_path in tqdm(phocal_train):
        if img_path.split('/')[0] == 'train':
            intrinsics = train_intrinsic[int(img_path.split('/')[1].split('_')[-1])-1]
        img_full_path = os.path.join(data_dir, img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_test_data(data_dir):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """
    real_test = open(os.path.join(data_dir, 'test_list_all.txt')).read().splitlines()
    # compute model size
    model_file_path = 'obj_models_small_size/phocal_test.pkl'
    models = {}
    with open(os.path.join(data_dir, model_file_path), 'rb') as f:
        models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)

    scale_factors = {}
    obj_list = os.listdir(os.path.join(data_dir,'obj_models_small_size'))
    for obj in obj_list:
        path_to_size = glob.glob(os.path.join(data_dir, 'obj_models_small_size',obj, '*.obj'))
        for inst_path in sorted(path_to_size):
            instance = os.path.basename(inst_path).split('.')[0]
            if instance in test_obj:
                x,y,z = calculate_xyz_bbox_size(inst_path)
                bbox_dims = np.array((x,y,z))
                scale_factors[instance] = np.linalg.norm(bbox_dims)
                
    valid_img_list = []
    for img_path in tqdm(real_test):
        if img_path.split('/')[0] == 'test':
            intrinsics = test_intrinsic[int(img_path.split('/')[1].split('_')[-1])-13]
        img_full_path = os.path.join(data_dir, img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        num_insts = len(instance_ids)
        ###
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        sizes = np.zeros((num_insts, 3))
        ###
        
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = np.array(class_ids)    # int list, 1 to 6
        gts['bboxes'] = bboxes    # np.array, [[y1, x1, y2, x2], ...]
        gts['instance_ids'] = instance_ids    # int list, start from 1
        gts['model_list'] = model_list    # str list, model id/name
        gts['size'] = sizes   # 3D size of NOCS model
        gts['scales'] = scales.astype(np.float32)    # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)    # np.array, T
        # gts['poses'] = poses.astype(np.float32)    # np.array
        # gts['handle_visibility'] = handle_visibility    # handle visibility of mug
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
        # write valid img list to file
        with open(os.path.join(data_dir, 'test'+'_list.txt'), 'w') as f:
            for img_path in valid_img_list:
                f.write("%s\n" % img_path)


if __name__ == '__main__':
    data_dir = '../data'
    # create list for all data
    # create_img_list(data_dir)
    # annotate dataset and re-write valid data to list
    # annotate_phocal_train(data_dir)
    annotate_test_data(data_dir)
