# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
from pathlib import Path
import pickle

import torch
import numpy as np
from scipy import spatial
import json
from PIL import Image

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, get_felz, fast_hist, per_class_iu
from lib.io3d import write_triangle_mesh, create_color_palette

from os import listdir
from os.path import isfile, join

# Import from project root
sys.path.insert(0, '../../lib')
from lib.constants.scannet_constants import *


class ScannetVoxelizationDataset(VoxelizationDataset):
    # added
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    NUM_IN_CHANNEL = 3
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    CLASS_LABELS_INSTANCE = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                             'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS_INSTANCE = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    IS_FULL_POINTCLOUD_EVAL = True
    depth_shape = (480,640)

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'scannetv2_train.txt',
        DatasetPhase.Val: 'scannetv2_val.txt',
        DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
        DatasetPhase.Test: 'scannetv2_test.txt'
    }

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.data.scannet_path
        felz_root = config.data.felz_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        data_paths = read_txt(os.path.join(data_root, 'splits', self.DATA_PATH_FILE[phase]))
        if phase == DatasetPhase.Train and config.data.train_file:
            data_paths = read_txt(os.path.join(data_root, 'splits', config.data.train_file))

        # data efficiency by sampling points
        self.sampled_inds = {}
        if config.data.sampled_inds and phase == DatasetPhase.Train:
            self.sampled_inds = torch.load(config.data.sampled_inds)
        
        
        felz_paths = [f for f in listdir(felz_root) if isfile(join(felz_root, f))]
        data_paths.sort()
        felz_paths = get_felz(data_paths,felz_paths)
        data_paths = [data_path + '.pth' for data_path in data_paths]

        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            felz_paths,
            data_root=data_root,
            felz_root=felz_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.data.ignore_label,
            return_transformation=config.data.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)

        # Load category weights for weighted CE and Focal
        self.category_weights = torch.ones(self.NUM_LABELS)
        category_weights_path = config.data.scannet_path + '/' + config.data.category_weights
        if os.path.isfile(category_weights_path):
            with open(category_weights_path, "rb") as input_file:
                category_weights = pickle.load(input_file)
                print('Loaded category weights for criterion {}'.format(category_weights_path))

            for cat_id, cat_value in category_weights.items():
                if cat_id > 0:
                    mapped_id = self.label_map[cat_id]
                    self.category_weights[mapped_id] = cat_value

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def get_classnames(self):
        return self.CLASS_LABELS

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def load_data(self, index):
        # loads all the data needed

        filepath = self.data_root / self.data_paths[index]
        felzpath = self.felz_root / self.felz_paths[index]
        scene_name = filepath.stem
        if self.sampled_inds:
            scene_name = self.get_output_id(index)
            mask = np.ones_like(labels).astype(np.bool)
            sampled_inds = self.sampled_inds[scene_name]
            mask[sampled_inds] = False
            labels[mask] = 0
            instances[mask] = 0
            
        pointcloud = torch.load(filepath)
        coords = pointcloud[0].astype(np.float32)
        feats = pointcloud[1].astype(np.float32)
        labels = pointcloud[2].astype(np.int32)
        instances = pointcloud[3].astype(np.int32)
        
        # Opening felz JSON file
        f = open(felzpath)
        data = json.load(f)
        segIndices = np.array(data["segIndices"],dtype=np.int32)
        segConnectivity = np.array(data["segConnectivity"],dtype=np.int32)
        f.close
        images = []
        camera_poses = []
        
        #maybe memory problem when loading many images
        paths = os.listdir("/mnt/hdd/scannet_2d_data/" + scene_name + "/color/")
        path2d = "/mnt/hdd/scannet_2d_data/" + scene_name
        for path in paths[:1]:
            img = Image.open(path2d + "/color/" + path)
            img = np.array(img)
            images.append(img)
            camera_poses.append(np.loadtxt(path2d + "/pose/" + path.replace(".jpg",".txt"), usecols=range(0,4), dtype=np.float32))
        images = np.stack(images, axis=0)
        images = np.moveaxis(images, [0, 1, 2, 3], [0, 3, 2, 1])
        camera_poses = np.stack(camera_poses, axis=0)
        
        color_intrinsics = np.loadtxt("/mnt/hdd/scannet_2d_data/" + scene_name + "/intrinsic/intrinsic_color.txt", usecols=range(0,4), dtype=np.float32)
        color_intrinsics = np.array([color_intrinsics[0][0],color_intrinsics[1][1],color_intrinsics[0][2],color_intrinsics[1][2]]).reshape(1,-1)

        return coords, feats, labels, instances, scene_name, images, camera_poses, color_intrinsics, segIndices, segConnectivity

    def get_original_pointcloud(self, coords, transformation, iteration):
        logging.info('===> Start testing on original pointcloud space.')
        data_path = self.data_paths[iteration]
        fullply_f = self.data_root / data_path
        query_xyz, _, query_label, _ = torch.load(fullply_f)

        coords = coords[:, 1:].numpy() + 0.5
        curr_transformation = transformation[0, :16].numpy().reshape(4, 4)
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
        coords = (np.linalg.inv(curr_transformation) @ coords.T).T

        # Run test for each room.
        from pykeops.numpy import LazyTensor
        # from pykeops.numpy.utils import IsGpuAvailable

        query_xyz = np.array(query_xyz)
        x_i = LazyTensor(query_xyz[:, None, :])  # x_i.shape = (1e6, 1, 3)
        y_j = LazyTensor(coords[:, :3][None, :, :].astype(np.float32))  # y_j.shape = ( 1, 2e6,3)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
        indKNN = D_ij.argKmin(1, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
        inds = indKNN[:, 0]
        return inds, query_xyz

    def save_prediction(self, coords, pred, transformation, iteration, save_dir):
        print('Running full pointcloud evaluation.')
        # if dataset.IGNORE_LABELS:
        #  decode_label_map = {}
        #  for k, v in dataset.label_map.items():
        #    decode_label_map[v] = k
        #  orig_pred = np.array([decode_label_map[x.item()] for x in orig_pred.cpu()], dtype=np.int)
        inds_mapping, xyz = self.get_original_pointcloud(coords, transformation, iteration)
        save = {'points': coords, 'mapping': inds_mapping, 'labels': pred}

        # Save prediciton in txt format for submission.
        room_id = self.get_output_id(iteration)
        torch.save(save, os.path.join(save_dir, room_id))
        # np.savetxt(f'{save_dir}/{room_id}.txt', ptc_pred, fmt='%i')

    def save_groundtruth(self, coords, gt, transformation, iteration, save_dir):
        save = {'points': coords, 'labels': gt}
        # Save prediciton in txt format for submission.
        room_id = self.get_output_id(iteration)
        torch.save(save, os.path.join(save_dir, room_id))


class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02


class Scannet200VoxelizationDataset(ScannetVoxelizationDataset):
    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_200
    CLASS_LABELS = CLASS_LABELS_200
    VALID_CLASS_IDS = VALID_CLASS_IDS_200

    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys()) + 1
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    CLASS_LABELS_INSTANCE = np.array([l for l in CLASS_LABELS if l not in PARENT_CLASS_SUPERCAT])
    VALID_CLASS_IDS_INSTANCE = np.array([i for i in VALID_CLASS_IDS if i not in VALID_PARENT_IDS_SUPERCAT])
    IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))


class Scannet200Voxelization2cmDataset(Scannet200VoxelizationDataset):
    VOXEL_SIZE = 0.02
