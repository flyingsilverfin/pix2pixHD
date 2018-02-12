### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
import os

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True 
opt.no_flip = True 
opt.instance_feat = True
opt.which_epoch = 'latest'
opt.continue_train = True

opt.instance_feat = True # put in features to steer generation
opt.name = 'label2city_512p_feat'
opt.dataroot = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/datasets/cityscapes/'
opt.checkpoints_dir = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/checkpoints/'
opt.results = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/results/'
opt.no_flip = True

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)




############ Initialize #########
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)



########### Encode features ###########
reencode = False
if reencode:
        features = {}
        for label in range(opt.label_nc):
                features[label] = np.zeros((0, opt.feat_num+1))
        for i, data in enumerate(dataset):    
            feat = model.module.encode_features(data['image'], data['inst'])
#            if 'jena_000034_000019' in data['path'][-1]:
#                print(data)
#                print(feat)
#                import sys
#                sys.exit(1)
            for label in range(opt.label_nc):
                features[label] = np.append(features[label], feat[label], axis=0) 
                
            print('%d / %d images' % (i+1, dataset_size))    
        save_name = os.path.join(save_path, name + '.npy')
        np.save(save_name, features)

############## Clustering ###########
n_clusters = opt.n_clusters
load_name = os.path.join(save_path, name + '.npy')
features = np.load(load_name).item()
from sklearn.cluster import KMeans
centers = {}
for label in range(opt.label_nc):
        feat = features[label]
        feat = feat[feat[:,-1] > 0.5, :-1]              
        if feat.shape[0]:
            n_clusters = min(feat.shape[0], opt.n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
            centers[label] = kmeans.cluster_centers_
            print(centers[label].shape)
        else:
            #centers[label] = np.array([0.0, 0.0, 0.0]) # something at all 
            pass
save_name = os.path.join(save_path, name + '_clustered_%03d.npy' % opt.n_clusters)
np.save(save_name, centers)
print('saving to %s' % save_name)
