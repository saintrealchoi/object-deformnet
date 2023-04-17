import os
import sys
import h5py
import glob
import numpy as np
import _pickle as cPickle
sys.path.append('/home/choisj/git/sj/phocal_object_deformnet/object-deformnet/lib')
from utils import sample_points_from_mesh, calculate_xyz_bbox_size

train_obj = {'box_leibniz_choco', 'box_principe', 'can_couscous', 'bottle_eres_inox', 'can_meatball', 'box_crunchy_muesli', 'cup_green_actys', 'box_koala', 'cutlery_spoon_2', 'glass_jar_big', 'remote_tv_white_quelle', 'teapot_wooden_color', 'cutlery_fork_3', 'cup_stanford', 'remote_grey', 'cutlery_fork_2', 'box_barilla', 'cup_white_whisker', 'glass_green_bottle', 'glass_beer_mug', 'cup_yellow_white_border', 'cutlery_knife_2', 'remote_aircon_chunghop', 'glass_ashtray_big', 'can_fanta', 'bottle_evian_frozen', 'cutlery_spoon_1', 'cup_yellow_handle', 'can_large_tuna', 'teapot_white_blue_cone_top', 'cutlery_knife_1'}
test_obj = {'remote_jaxster', 'can_monster', 'bottle_evian_red', 'teapot_big_white_floral', 'remote_toy', 'can_sheba_cat', 'remote_infini_fun', 'bottle_85_alcool', 'teapot_white_was_brand', 'teapot_brown_chinese', 'cup_green_actys', 'cutlery_spatula', 'bottle_soupline', 'can_tuna_salad', 'cutlery_scissors', 'glass_small_5', 'cup_new_york_big', 'cutlery_fork_3', 'box_proschmelz', 'can_corned_beef', 'bottle_dettol_washing_machine', 'glass_jar_landliebe', 'glass_green_bottle', 'cutlery_knife_2', 'cutlery_fork_1', 'cutlery_peeler', 'glass_cup_small_2', 'cup_plastic_green_flowers', 'box_special_k', 'cup_red_heart', 'box_antikalk', 'glass_cocktail', 'teapot_ambition_brand'}

def save_nocs_model_to_file(obj_model_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5

    """
    # PhoCaL
    phocal_train = {}
    phocal_test = {}
    for subset in os.listdir(obj_model_dir):
        inst_list = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in inst_list:
            instance = os.path.basename(inst_path).split('.')[0]
            x,y,z = calculate_xyz_bbox_size(inst_path)
            bbox_dims = np.array((x,y,z))
            scale = np.linalg.norm(bbox_dims)
            model_points = sample_points_from_mesh(inst_path, 2048, fps=True, ratio=3)
            model_points /= scale
            if instance in train_obj:
                phocal_train[instance] = model_points
            else:
                phocal_test[instance] = model_points
    with open(os.path.join(obj_model_dir, '{}.pkl'.format('phocal_train')), 'wb') as f:
        cPickle.dump(phocal_train, f)
    with open(os.path.join(obj_model_dir, '{}.pkl'.format('phocal_test')), 'wb') as f:
        cPickle.dump(phocal_test, f)


def save_model_to_hdf5(obj_model_dir, n_points, fps=False, include_distractors=False, with_normal=False):
    """ Save object models (point cloud) to HDF5 file.
        Dataset used to train the auto-encoder.
        Only use models from ShapeNetCore.
        Background objects are not inlcuded as default. We did not observe that it helps
        to train the auto-encoder.

    """
    cat_dict = {'bottle':1,'box':2,'can':3,'cup':4,'cutlery':5,'glass':6,'remote':7,'teapot':8}
    # read all the paths to models
    print('Sampling points from mesh model ...')
        
    train_data = np.zeros((3000, n_points, 3), dtype=np.float32)
    val_data = np.zeros((500, n_points, 3), dtype=np.float32)
    train_label = []
    val_label = []
    train_count = 0
    val_count = 0
    # Phocal
    for subset in os.listdir(obj_model_dir):
        path_to_mesh_models = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in sorted(path_to_mesh_models):
            instance = os.path.basename(inst_path).split('.')[0]
            catId =  cat_dict[inst_path.split('/')[-2]]
            model_points = sample_points_from_mesh(inst_path, n_points, with_normal, fps=fps, ratio=2)
            x,y,z = calculate_xyz_bbox_size(inst_path)
            bbox_dims = np.array((x,y,z))
            model_points /= np.linalg.norm(bbox_dims)
            # model_data[model_count] = model_points
            # model_label.append(catId)
            # model_count +=1
            if instance in train_obj:
                train_data[train_count] = model_points
                train_label.append(catId)
                train_count += 1
            else:
                val_data[val_count] = model_points
                val_label.append(catId)
                val_count += 1

    num_train_instances = len(train_label)
    num_val_instances = len(val_label)
    # num_model_instances = len(model_label)
    assert num_train_instances == train_count
    assert num_val_instances == val_count
    # assert num_model_instances == model_count
    # model_data = model_data[:num_model_instances]
    train_data = train_data[:num_train_instances]
    val_data = val_data[:num_val_instances]
    # model_label = np.array(model_label, dtype=np.uint8)
    train_label = np.array(train_label, dtype=np.uint8)
    val_label = np.array(val_label, dtype=np.uint8)
    print('{} shapes found in train dataset'.format(num_train_instances))
    print('{} shapes found in val dataset'.format(num_val_instances))
    # print('{} shapes found in phocal dataset'.format(num_model_instances))
    # write to HDF5 file
    print('Writing data to HDF5 file ...')
    if with_normal:
        filename = 'ShapeNetCore_{}_with_normal.h5'.format(n_points)
    else:
        filename = 'ShapeNetCore_{}.h5'.format(n_points)
    hfile = h5py.File(os.path.join(obj_model_dir, filename), 'w')
    # train_dataset = hfile.create_group('phocal')
    # train_dataset.attrs.create('len', num_model_instances)
    # train_dataset.create_dataset('data', data=model_data, compression='gzip', dtype='float32')
    # train_dataset.create_dataset('label', data=model_label, compression='gzip', dtype='uint8')
    train_dataset = hfile.create_group('train')
    train_dataset.attrs.create('len', num_train_instances)
    train_dataset.create_dataset('data', data=train_data, compression='gzip', dtype='float32')
    train_dataset.create_dataset('label', data=train_label, compression='gzip', dtype='uint8')
    val_dataset = hfile.create_group('val')
    val_dataset.attrs.create('len', num_val_instances)
    val_dataset.create_dataset('data', data=val_data, compression='gzip', dtype='float32')
    val_dataset.create_dataset('label', data=val_label, compression='gzip', dtype='uint8')
    hfile.close()


if __name__ == '__main__':
    obj_model_dir = '/data/PhoCAL_release/obj_models_small_size'
    # Save ground truth models for training deform network
    save_nocs_model_to_file(obj_model_dir)
    # Save models to HDF5 file for training the auto-encoder.
    save_model_to_hdf5(obj_model_dir, n_points=4096, fps=False)
    # # Save nmodels to HDF5 file, which used to generate mean shape.
    save_model_to_hdf5(obj_model_dir, n_points=2048, fps=True)

    # import random
    # import open3d as o3d
    # for file in ['camera_train.pkl', 'camera_val.pkl', 'real_train.pkl', 'real_test.pkl']:
    #     with open(os.path.join(obj_model_dir, file), 'rb') as f:
    #         obj_models = cPickle.load(f)
    #     instance = random.choice(list(obj_models.keys()))
    #     model_points = obj_models[instance]
    #     print('Diameter: {}'.format(np.linalg.norm(2*np.amax(np.abs(model_points), axis=0))))
    #     color = np.repeat(np.array([[1, 0, 0]]), model_points.shape[0], axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(model_points)
    #     pcd.colors = o3d.utility.Vector3dVector(color)
    #     # visualization: camera coordinate frame
    #     points = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    #     lines = [[0, 1], [0, 2], [0, 3]]
    #     colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(points)
    #     line_set.lines = o3d.utility.Vector2iVector(lines)
    #     line_set.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.visualization.draw_geometries([pcd, line_set])
