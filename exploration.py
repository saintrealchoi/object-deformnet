import os
import json
import pickle
import numpy as np

def main():
    dataset_path = 'data'
    ttl_train = 0
    ttl_test = 0
    for data in sorted(os.listdir(dataset_path)):
        if not data.startswith('sequence'):
            continue
        
        data_len = os.path.join(dataset_path,data,'rgb')
        npz_file = os.path.join(dataset_path,data,'train_test_split.npz')
        train_npy = np.load(npz_file)['train_idxs']
        train_npy = len(train_npy)
        test_npy = np.load(npz_file)['test_idxs']
        test_npy = len(test_npy)
        
        print(data,":  / total : ",len(os.listdir(data_len))," / train : ",train_npy," / test : ",test_npy)
        ttl_train += train_npy
        ttl_test += test_npy
        rgb_gt = os.path.join(dataset_path,data,'rgb_scene_gt.json')
        with open(rgb_gt,'r') as f:
            rgb_gt = json.load(f)
        
        for i in rgb_gt:
            for j in rgb_gt[i]:
                break
            break
    print(ttl_train)
    print(ttl_test)
    # REAL275 dataset
    # data = '/data/NOCS/Real/train/scene_1/0000_label.pkl'
    # with open(data,'rb') as f:
    #     test = pickle.load(f)
    # print(test)
    
    

if __name__ == '__main__':
    main()