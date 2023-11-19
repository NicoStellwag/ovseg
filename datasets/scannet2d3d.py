import glob
import logging
import os
import sys
import warnings
import numbers
from pathlib import Path
import pickle
import random

from PIL import Image
import numpy as np
from scipy import spatial, ndimage, misc
import pandas as pd
import torch
from scipy.spatial import KDTree
import open3d as o3d
import MinkowskiEngine as ME
import MinkowskiEngine as ME

from datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from datasets.preprocessing.utils import load_matrix_from_txt, load_labels
from utils.pc_utils import read_plyfile, save_point_cloud
from utils.utils import read_txt, fast_hist, per_class_iu

from datasets.scannet200.scannet200_constants import *
from datasets.scannet200.scannet200_splits import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200


class Scannet_2D3D_Dataset(VoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_20
    CLASS_LABELS = CLASS_LABELS_20
    VALID_CLASS_IDS = VALID_CLASS_IDS_20
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train.txt',
        DatasetPhase.Val: 'val.txt',
        DatasetPhase.TrainVal: 'trainval.txt',
        DatasetPhase.Test: 'test.txt',
        DatasetPhase.Debug: 'debug.txt',
        DatasetPhase.Clean: 'clean.txt'
    }

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 image_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 verbose=False,
                 phase=DatasetPhase.Train,
                 data_root=None):

        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        self.phase = phase

        if data_root is None:
            # Load chunk paths for dataset
            data_root = config.data.scannet_path
            data_scenes = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
            data_paths = []
            for s_name in data_scenes:
                data_paths += sorted(glob.glob(os.path.join(data_root, config.data.chunks_folder) + f'/{s_name}*'))
            self.data_paths = np.array(data_paths)

            logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))

        # Init DatasetBase
        super().__init__(
            self.data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.data.ignore_label,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
            cache=cache)

        # Save Image transforms
        self.image_transform = image_transform

        # Load dataframe with label map
        labels_pd = pd.read_csv(config.data.scannet_path + '/' + config.data.label_mapping, sep='\t', header=0)
        labels_pd.loc[labels_pd.raw_category == 'stick', ['category']] = 'object'
        labels_pd.loc[labels_pd.category == 'wardrobe ', ['category']] = 'wardrobe'
        self.labels_pd = labels_pd

        # Create label map
        label_map = {}
        for index, row in labels_pd.iterrows():
            id = row['id']
            nyu40id = row['nyu40id']
            if nyu40id in self.VALID_CLASS_IDS:
                scannet20_index = self.VALID_CLASS_IDS.index(nyu40id)
                label_map[id] = scannet20_index
            else:
                label_map[id] = self.ignore_mask

        # Add ignore
        label_map[0] = self.ignore_mask
        label_map[self.ignore_mask] = self.ignore_mask
        self.label_map = label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])
        self.NUM_LABELS = len(self.VALID_CLASS_IDS)

        # Precompute a mapping from ids to categories
        self.id2cat_name = {}
        self.cat_name2id = {}
        for id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            self.id2cat_name[id] = cat_name
            self.cat_name2id[cat_name] = id

        # 2D image shape from config
        scale = self.config.net_2d.downsample_ratio
        self.depth_shape = tuple(int(scale * dim) for dim in self.config.net_2d.image_resolution)
        self.pil_depth_shape = tuple(reversed(self.depth_shape))
        self.depth_image_treshold = config.data.depth_image_treshold / self.VOXEL_SIZE

        self.category_weights = None

        # Everything is head
        self.head_ids = np.arange(self.NUM_LABELS)
        self.common_ids = []
        self.tail_ids = []

        # Determine max image num for chunks
        self.image_num = self.config.net_2d.max_image_num if phase in [DatasetPhase.Train] else self.config.net_2d.test_image_num

        # If we need textual pretraining load CLIP features for it
        if self.config.optimizer.embedding_phase:
            language_features_path = os.path.join(config.data.scannet_path, config.data.language_features_path)
            if os.path.isfile(language_features_path):
                with open(language_features_path, 'rb') as f:
                    self.loaded_text_features = pickle.load(f)
                logging.info(f"Loaded language grounding features from: {language_features_path}")
            else:
                logging.info(f"Can't find file for language grounding: {language_features_path}")

    def load_scene_data(self, index):

        with open(self.data_paths[index], 'rb') as handle:
            try:
                scene_dict = pickle.load(handle)
            except:
                print(f"An exception occurred with input chunk {self.data_paths[index]}")
                with open(self.data_paths[index - 1], 'rb') as handle1:
                    scene_dict = pickle.load(handle1)

        scene_name = scene_dict['scene_name']
        target_frames = np.array(scene_dict['greedy_frame_ids'])
        if self.config.net_2d.stochastic_sampling:
            max_frames = len(target_frames)
            frame_ids = np.random.choice(max_frames, size=self.image_num, replace=False)
            target_frames = target_frames[frame_ids]
        else:
            target_frames = target_frames[:self.image_num] if self.image_num != -1 else target_frames

        # Load intrinsics info
        info_file = os.path.join(self.config.data.scannet_2d_path, f'{scene_name}', f'{scene_name}.txt')
        info_dict = {}
        with open(info_file) as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for line in f:
                    (key, val) = line.split(" = ")
                    info_dict[key] = np.fromstring(val, sep=' ')
        if 'axisAlignment' in info_dict and self.config.data.align_scenes:
            axis_alignment = info_dict['axisAlignment'].reshape(4, 4)
        else:
            axis_alignment = np.identity(4)

        # Load intrinsics
        orig_color_shape = np.array([info_dict['colorHeight'], info_dict['colorWidth']])
        color_scale = [self.depth_shape[0] / orig_color_shape[0], self.depth_shape[1] / orig_color_shape[1]]

        color_intrinsics = np.array([info_dict['fx_color'] * color_scale[0],
                                     info_dict['fy_color'] * color_scale[1],
                                     info_dict['mx_color'] * color_scale[0],
                                     info_dict['my_color'] * color_scale[1]])

        # Load rgb images
        RGB_PATH = self.config.data.scannet_2d_path + f'/{scene_name}/color/'
        rgb_paths = sorted(glob.glob(RGB_PATH + '*.jpg'))
        rgb_paths = [rgb_paths[i] for i in target_frames]
        rgb_images = np.array([self.load_image(rgb_path, self.pil_depth_shape) for rgb_path in rgb_paths])

        # Load depth images
        DEPTH_PATH = self.config.data.scannet_2d_path + f'/{scene_name}/depth/'
        depth_paths = sorted(glob.glob(DEPTH_PATH + '*.png'))
        depth_paths = [depth_paths[i] for i in target_frames]
        depth_images = np.array([self.load_image(depth_path, self.pil_depth_shape, label=True) for depth_path in depth_paths])

        # Load label images
        LABEL_PATH = self.config.data.scannet_2d_path + f'/{scene_name}/label/'
        label_paths = sorted(glob.glob(LABEL_PATH + '*.png'))
        label_paths = [label_paths[i] for i in target_frames]
        labels_2d = np.array([self.load_image(label_path, self.pil_depth_shape, label=True) for label_path in label_paths])

        # Load camera poses
        POSE_PATH = self.config.data.scannet_2d_path + f'/{scene_name}/pose/'
        pose_paths = sorted(glob.glob(POSE_PATH + '*.txt'))
        pose_paths = [pose_paths[i] for i in target_frames]
        poses_2d = np.array([axis_alignment @ load_matrix_from_txt(pose_path) for pose_path in pose_paths])

        # we need to pad if not enough frames for chunk
        if len(rgb_paths) < self.image_num and self.image_num != -1:
            missing_dims = self.image_num - len(rgb_paths)

            if len(rgb_paths) == 0:
                rgb_images = np.zeros((missing_dims, *self.depth_shape), dtype=np.int64)
                depth_images = np.zeros((missing_dims, *self.depth_shape), dtype=np.int64)
                labels_2d = np.zeros((missing_dims, *self.depth_shape), dtype=np.int64)
                poses_2d = np.tile(np.eye(4)[np.newaxis, :, :], (missing_dims, 1, 1)).astype(float)
            else:
                rgb_images = np.append(rgb_images, np.zeros((missing_dims, *self.depth_shape, 3), dtype=rgb_images.dtype), axis=0)
                depth_images = np.append(depth_images, np.zeros((missing_dims, *self.depth_shape), dtype=depth_images.dtype), axis=0)
                labels_2d = np.append(labels_2d, np.zeros((missing_dims, *self.depth_shape), dtype=labels_2d.dtype), axis=0)
                poses_2d = np.append(poses_2d, np.tile(np.eye(4)[np.newaxis, :, :], (missing_dims, 1, 1)).astype(float), axis=0)

        # Load sampled point cloud too if necessary
        mesh_labels = scene_dict['mesh_labels']
        mesh_instances = scene_dict['mesh_instances']
        mesh_vertices = scene_dict['mesh_vertices']
        vertex_colors = scene_dict['mesh_vertex_colors']
        surface_area = scene_dict['surface_area']
        mesh_faces = scene_dict['mesh_faces']
        vertex_normals = scene_dict['mesh_normals']
        instance_info = scene_dict['instance_info'] if 'instance_info' in scene_dict else None

        return rgb_images, depth_images, labels_2d, poses_2d, color_intrinsics, scene_name, mesh_labels, mesh_instances, mesh_vertices, vertex_colors, vertex_normals, mesh_faces, surface_area, instance_info


    def __getitem__(self, index):

        rgb_images, depth_images, labels_2d, poses_2d, color_intrinsics, scene_name, *scene_mesh = self.load_scene_data(index)
        mesh_labels, mesh_instances, mesh_vertices, vertex_colors, vertex_normals, mesh_faces, surface_area, instance_info = scene_mesh
        vertex_colors = np.floor(vertex_colors * 255.).astype(float)

        # Strip images to use max frames/chunk
        image_num = self.image_num if self.image_num > 0 else 10000  # random big number to use all images
        rgb_images = rgb_images[:image_num]
        depth_images = depth_images[:image_num]
        labels_2d = labels_2d[:image_num]
        poses_2d = poses_2d[:image_num]

        indexer = np.arange(mesh_vertices.shape[0])

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            mesh_vertices, vertex_colors, indexer = self.prevoxel_transform(mesh_vertices, vertex_colors, indexer)

        # For some networks, making the network invariant to even, odd coords is important. Random translation
        if self.augment_data:
            rand_shift = (np.random.rand(3) * 2).astype(mesh_vertices.dtype)
            mesh_vertices += rand_shift - 0.5  # the extra is the voxelization shift that comes with clipping discretization
            poses_2d[:, :3, 3] += rand_shift - 0.5

        # Voxelize
        coords, colors, indexer, transformations = self.voxelizer.voxelize(mesh_vertices, vertex_colors, indexer)

        # Transform poses with voxelization scale and rotation
        scale_transform, rotation_transform = transformations
        poses_2d[:, :, 3] = poses_2d[:, :, 3] @ (scale_transform @ rotation_transform).T
        poses_2d[:, :3, :3] = rotation_transform[:3, :3] @ poses_2d[:, :3, :3]
        depth_images = depth_images * scale_transform[0, 0] / 1000.  # transform from millimeter to voxel dists
        voxel_transform = rotation_transform @ scale_transform

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, colors, indexer = self.input_transform(coords, colors, indexer)
        if self.target_transform is not None:
            coords, colors, indexer = self.target_transform(coords, colors, indexer)

        # Drop values stripped due to augmentation and voxelization, then map to output dim
        labels = mesh_labels[indexer]
        instances = mesh_instances[indexer]

        # Apply image transformation if requested
        if self.image_transform is not None:
            rgb_images = torch.from_numpy(rgb_images / 255.).permute(0, 3, 1, 2).contiguous().float()
            rgb_images = self.image_transform(rgb_images)

        if self.IGNORE_LABELS is not None:
            labels = self.label_mapper(labels)
            labels_2d = self.label_mapper(labels_2d)

        # If querying is necessary sample points, colors and labels
        if self.config.data.sampled_points > 0:

            num_crop_points = mesh_labels.shape[0]
            surface_area = surface_area if np.isfinite(surface_area) else 1.
            sample_num = self.config.data.sampled_points if self.config.net_implicit.fixed_sample_num else round(self.config.data.sampled_points * surface_area)

            if num_crop_points == 0:
                sampled_points = np.zeros((sample_num, 3))
                sampled_colors = np.zeros((sample_num, 3))
                sampled_normals = np.zeros((sample_num, 3))
                sampled_labels = np.ones(sample_num) * self.config.data.ignore_label
                print(self.data_paths[index], 'zero points in chunk...')

            else:
                if self.phase == DatasetPhase.Debug:
                    np.random.seed(0)
                sampled_inds = np.random.choice(num_crop_points, sample_num, replace=(num_crop_points < sample_num))
                sampled_points = mesh_vertices[sampled_inds]
                sampled_colors = vertex_colors[sampled_inds]
                sampled_normals = vertex_normals[sampled_inds] if vertex_normals.shape[0] > 0 else np.ones((sample_num, 3))
                sampled_labels = mesh_labels[sampled_inds]
                sampled_instances = mesh_instances[sampled_inds]

                # Apply additive sampling for tail categories if requested
                if isinstance(self.config.data.category_weights, numbers.Number):
                    is_tailcat = np.vectorize(
                        lambda cat_id: self.CLASS_LABELS[self.label_map[cat_id]] in TAIL_CATS_SCANNET_200)
                    tail_cat_points = is_tailcat(mesh_labels)
                    sampled_tail_cats = is_tailcat(sampled_labels)
                    num_tail_cat_points = sampled_tail_cats.sum()
                    all_tail_cat_points = tail_cat_points.sum()

                    targeted_sampled_inds = np.random.choice(np.arange(num_crop_points)[tail_cat_points],
                                                             min(num_tail_cat_points * self.config.data.category_weights,
                                                                 all_tail_cat_points),
                                                             replace=False)
                    targeted_sampled_points = mesh_vertices[targeted_sampled_inds]
                    targeted_sampled_colors = vertex_colors[targeted_sampled_inds]
                    targeted_sampled_normals = vertex_normals[targeted_sampled_inds]
                    targeted_sampled_labels = mesh_labels[targeted_sampled_inds]
                    targeted_sampled_instances = mesh_instances[targeted_sampled_inds]

                    sampled_points = np.concatenate((sampled_points, targeted_sampled_points), axis=0)
                    sampled_colors = np.concatenate((sampled_colors, targeted_sampled_colors), axis=0)
                    sampled_normals = np.concatenate((sampled_normals, targeted_sampled_normals), axis=0)
                    sampled_labels = np.concatenate((sampled_labels, targeted_sampled_labels), axis=0)
                    sampled_instances = np.concatenate((sampled_instances, targeted_sampled_instances), axis=0)

                # Apply voxelization transform to sampled points
                sampled_colors = sampled_colors / 255.
                sampled_labels = self.label_mapper(sampled_labels)
                homo_coords = np.hstack((sampled_points, np.ones((sampled_points.shape[0], 1), dtype=sampled_points.dtype)))
                sampled_points = homo_coords @ voxel_transform.T[:, :3]

                # rotate normals as well
                homo_normals = np.hstack((sampled_normals, np.ones((sampled_normals.shape[0], 1), dtype=sampled_normals.dtype)))
                sampled_normals = homo_normals @ rotation_transform.T[:, :3]
        else:
            sampled_points = None
            sampled_colors = None
            sampled_normals = None
            sampled_labels = None
            sampled_instances = None

        # we want to precompute NNs with the sampled points for the transformer decoder
        if self.config.net_implicit.mlp_mode == 'point_transformer':
            # Sample neighbors from full cloud
            NNs = self.nn_sampling(sampled_points)
        else:
            NNs = np.zeros((sampled_points.shape[0], 1)).astype(int)

        return_args = [coords, colors, labels, instances, rgb_images, depth_images, labels_2d, poses_2d, color_intrinsics, scene_name, sampled_points, sampled_colors, sampled_normals, sampled_labels, sampled_instances, NNs, voxel_transform.astype(np.float32)]

        # we need the mesh params for rasterizing the sampling surface points
        if self.config.train.supervise_with_2d:
            # first transform the full set of verts, then append to returns
            homo_verts = np.hstack((mesh_vertices, np.ones((mesh_vertices.shape[0], 1), dtype=mesh_vertices.dtype)))
            mesh_vertices = homo_verts @ voxel_transform.T[:, :3]
            return_args.append(mesh_vertices)
            return_args.append(mesh_faces)

        if 'instseg' in self.config.data.dataset.lower():
            return_args.append(instance_info)

        return tuple(return_args)

    def load_image(self, path, target_shape, label=False):
        if label:
            image = Image.open(path).resize(target_shape, Image.NEAREST)
        else:
            image = Image.open(path).convert('RGB').resize(target_shape, Image.BILINEAR)

        return np.array(image).astype(int)

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def get_classids(self):
        return self.VALID_CLASS_IDS

    def get_classnames(self):
        return self.CLASS_LABELS

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def test_pointcloud(self, pred_dir, num_labels):
        print('Running full pointcloud evaluation.')
        eval_path = os.path.join(pred_dir, 'fulleval')
        os.makedirs(eval_path, exist_ok=True)
        # Join room by their area and room id.
        # Test independently for each room.
        sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
        hist = np.zeros((num_labels, num_labels))
        for i, data_path in enumerate(self.data_paths):
            room_id = self.get_output_id(i)
            pred = np.load(os.path.join(pred_dir, 'pred_%04d_%02d.npy' % (i, 0)))

            # save voxelized pointcloud predictions
            save_point_cloud(
                np.hstack((pred[:, :3], np.array([self.SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))),
                f'{eval_path}/{room_id}_voxel.ply',
                verbose=False)

            fullply_f = self.data_root / data_path
            query_pointcloud = read_plyfile(fullply_f)
            query_xyz = query_pointcloud[:, :3]
            query_label = query_pointcloud[:, -1]
            # Run test for each room.
            pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
            _, result = pred_tree.query(query_xyz)
            ptc_pred = pred[result, 3].astype(int)
            # Save prediciton in txt format for submission.
            np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
            # Save prediciton in colored pointcloud for visualization.
            save_point_cloud(
                np.hstack((query_xyz, np.array([self.SCANNET_COLOR_MAP[i] for i in ptc_pred]))),
                f'{eval_path}/{room_id}.ply',
                verbose=False)
            # Evaluate IoU.
            if self.IGNORE_LABELS is not None:
                ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
                query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
            hist += fast_hist(ptc_pred, query_label, num_labels)
        ious = per_class_iu(hist) * 100
        print('mIoU: ' + str(np.nanmean(ious)) + '\n'                                                 'Class names: ' + ', '.join(self.CLASS_LABELS) + '\n'
                                                                                                  'IoU: ' + ', '.join(
            np.round(ious, 2).astype(str)))

    def nn_sampling(self, coords):

        if self.config.net_implicit.nn_type == 'radius':
            tree = KDTree(coords, leafsize=8)
            NNs = tree.query_ball_point(coords, r=self.config.net_implicit.nn_radius, workers=4)
            NNs = np.array([random.sample(nn, self.config.net_implicit.nearest_neighbours) if len(nn) > self.config.net_implicit.nearest_neighbours else np.random.choice(nn, size=self.config.net_implicit.nearest_neighbours) for nn in NNs])

        else:
            tree = KDTree(coords, leafsize=8)  # input_pickle['kd_tree']
            NNs = tree.query(coords, k=self.config.net_implicit.nearest_neighbours + 1)[1][:, 1:]  # we are only interested in the index and without itself

        return NNs

class Scannet_2D3D_2cmDataset(Scannet_2D3D_Dataset):
    VOXEL_SIZE = 0.02

class Scannet_2D3D_10cmDataset(Scannet_2D3D_Dataset):
    VOXEL_SIZE = 0.10

class Scannet_2D3D_20cmDataset(Scannet_2D3D_Dataset):
    VOXEL_SIZE = 0.20

class Scannet_2D3D_40cmDataset(Scannet_2D3D_Dataset):
    VOXEL_SIZE = 0.40

class Scannet200_2D3D_Dataset(Scannet_2D3D_Dataset):

    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_LONG
    CLASS_LABELS = CLASS_LABELS_200
    VALID_CLASS_IDS = VALID_CLASS_IDS_200

    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys()) + 1
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 image_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 verbose=False,
                 phase=DatasetPhase.Train):

        super(Scannet200_2D3D_Dataset, self).__init__(config,
                                                     prevoxel_transform=prevoxel_transform,
                                                     input_transform=input_transform,
                                                     target_transform=target_transform,
                                                     image_transform=image_transform,
                                                     augment_data=augment_data,
                                                     elastic_distortion=elastic_distortion,
                                                     cache=cache,
                                                     verbose=verbose,
                                                     phase=phase)

        # map labels not evaluated to ignore_label
        label_map = {}
        inverse_label_map = {}
        n_used = 0
        for l in range(max(SCANNET_COLOR_MAP_LONG.keys()) + 1):
            if l in self.IGNORE_LABELS:
                label_map[l] = self.ignore_mask
            else:
                label_map[l] = n_used
                inverse_label_map[n_used] = l
                n_used += 1
        label_map[self.ignore_mask] = self.ignore_mask
        inverse_label_map[self.ignore_mask] = 0
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])

        self.NUM_LABELS = len(self.VALID_CLASS_IDS)

        # Load category weights for weighted CE, Focal and targeted sampling
        # If category weights is constant - use that for tail cats, besides load weighting
        self.category_weights = torch.ones(self.NUM_LABELS)
        if isinstance(config.data.category_weights, str):
            category_weights_path = config.data.scannet_path + '/' + config.data.category_weights
            if os.path.isfile(category_weights_path):
                with open(category_weights_path, "rb") as input_file:
                    category_weights = pickle.load(input_file)
                    if verbose:
                        print('Loaded category weights for CE {}'.format(category_weights_path))

                for cat_id, cat_value in category_weights.items():
                    if cat_id > 0:
                        mapped_id = self.label_map[cat_id]
                        self.category_weights[mapped_id] = cat_value
        else:
            for cat_id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
                if cat_name in TAIL_CATS_SCANNET_200:
                    mapped_id = self.label_map[cat_id]
                    self.category_weights[mapped_id] = config.data.category_weights

        # Calculate head-common-tail ids
        self.head_ids = []
        self.common_ids = []
        self.tail_ids = []
        for scannet_id, scannet_cat in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            if scannet_cat in HEAD_CATS_SCANNET_200:
                self.head_ids += [self.label_map[scannet_id]]
            elif scannet_cat in COMMON_CATS_SCANNET_200:
                self.common_ids += [self.label_map[scannet_id]]
            elif scannet_cat in TAIL_CATS_SCANNET_200:
                self.tail_ids += [self.label_map[scannet_id]]


class Scannet200_2D3D_2cmDataset(Scannet200_2D3D_Dataset):
    VOXEL_SIZE = 0.02

class Scannet200_2D3D_10cmDataset(Scannet200_2D3D_Dataset):
    VOXEL_SIZE = 0.1

