import glob
import os
import sys
import hydra
#os.chdir("/home/luebberstedt/ovseg")
#sys.path.append(os.getcwd())

from sklearn.decomposition import PCA

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh
import torch.nn.functional as F

os.environ["HYDRA_FULL_ERROR"] = "1"
from utils.cuda_utils.raycast_image import Project2DFeaturesCUDA

from MinkowskiEngine import SparseTensor
import pyviz3d.visualizer as vis

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# Define hyperparameters here
max_number_of_instances = 20
eps = 1e-5
visualize = True

def normalize_mat(A, eps=1e-5):
    A -= np.min(A[np.nonzero(A)]) if np.any(A > 0) else 0
    A[A < 0] = 0.0
    A /= A.max() + eps
    return A


def get_affinity_matrix(
    feats, tau=0.15, eps=1e-5, normalize_sim=True, similarity_metric="cos"
):
    # get affinity matrix via measuring patch-wise cosine similarity

    if not isinstance(feats, tuple):  # single modality feature
        feats_a = F.normalize(feats, p=2, dim=-1).detach()#.cpu()
        feats_b = feats_a.T
        A = torch.matmul(feats_a,feats_b)
        # A = (
        #     cosine_similarity(feats_a[None,:,:], feats_b[:,None,:], dim=-1)
        #     #cosine_similarity(feats_a, feats_b,-1)
        #     if similarity_metric == "cos"
        #     else l2_sim(feats_a, feats_b)
        # )

        A = A.cpu().numpy()
        A = normalize_mat(A) if normalize_sim else A
    else:  # multi-modality feature, average the attention scores from both modalities
        # Calculate both attentions
        feats_a, feats_b = feats
        feats_a = F.normalize(feats_a, p=2, dim=-1)
        feats_b = F.normalize(feats_b, p=2, dim=-1)
        A_a, A_b = feats_a @ feats_a.T, feats_b @ feats_b.T

        # Normalize both attentions if requested
        A_a, A_b = A_a.cpu().numpy(), A_b.cpu().numpy()
        A_a = normalize_mat(A_a) if normalize_sim else A_a
        A_b = normalize_mat(A_b) if normalize_sim else A_b

        # Combine attentions
        A = (A_a + A_b) / 2

    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=0)
    D = np.diag(d_i)
    return A, D


def get_masked_affinity_matrix(painting, feats, mask):
    # mask out affinity matrix based on the painting matrix
    # painting matrix is basically an aggregated map of previous fgs
    # will be used as a multiplier to mask out features

    num_segment, dim = feats.shape if not isinstance(feats, tuple) else feats[0].shape
    painting = painting.view(num_segment, 1) + mask.view(num_segment, 1)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    if not isinstance(feats, tuple):
        feats = (1 - painting) * feats.clone()
    else:
        feats = ((1 - painting) * feats[0].clone(), (1 - painting) * feats[1].clone())
    return feats, painting.squeeze()


def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    # This is the solution of the normalized NCut algorithm
    # to determine 2 most connected parts of the similarity graph

    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec


def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition


def demean(feature_map, dim=-1):
    """removes mean of tensor channels"""
    mu = torch.mean(feature_map, dim=dim, keepdim=True)
    demeaned = -mu + feature_map
    return demeaned, mu


def whiten_feats(feature_map):
    """zca whiten each feature map within individual image in batch"""
    feature_map = F.normalize(feature_map, p=2, dim=-1)
    y = feature_map.T.unsqueeze(0) + 10e-8
    y, mu = demean(y)
    N = y.shape[-1]
    cov = torch.einsum("bcx, bdx -> bcd", y, y) / (N - 1)  # compute covs along batch
    u, lambduh, _ = torch.svd(cov)
    lambduh_inv_sqrt = torch.diag_embed(lambduh ** (-0.5))
    zca_whitener = torch.einsum(
        "nab, nbc, ncd -> nad", u, lambduh_inv_sqrt, u.transpose(-2, -1)
    )
    z = torch.einsum("bac, bcx -> bax", zca_whitener, y)

    white = (mu + z).squeeze().T
    white /= torch.abs(white).max()
    return white


def separate_segments(
    bipartition, second_smallest_vec, unique_segments, seg_connectivity, mode="max"
):
    # precompute connectivity dict
    connectivity_dict = {}
    for s_id in unique_segments:
        connectivity_dict[s_id.item()] = set(
            (seg_connectivity[seg_connectivity[:, 0] == s_id, 1].cpu().numpy())
        )

    # Separate non-connected regions
    curr_instances = []  # the fused blobs
    curr_mask_segment_ids = unique_segments[bipartition].cpu().numpy()

    # Iterate over all segments in query mask
    for c in curr_mask_segment_ids:
        # check if any set contains one of it's neighbours
        # if multiple contains, wer should merge those cells
        neighbour_segments = connectivity_dict[c.item()]
        last_fused_match = -1
        merged = False

        # iterate over all past blobs, associate and potentially merge if it was a bridge segment
        fused_id = 0
        while fused_id < len(curr_instances):
            fused_segments = curr_instances[fused_id]
            if len(neighbour_segments.intersection(fused_segments)) != 0:
                merged = True
                fused_segments.add(c)
                if last_fused_match != -1:
                    # merge with the previous segment, then delete
                    curr_instances[last_fused_match] = curr_instances[
                        last_fused_match
                    ].union(fused_segments)
                    curr_instances.pop(fused_id)
                else:
                    last_fused_match = fused_id

            fused_id += 1

        # add as new segment if never associated with others
        if not merged:
            curr_instances += [set([c])]

    # Get the one, where seed is in the segment
    curr_instances = np.array(curr_instances)

    if mode == "max":
        seed = np.argmax(second_smallest_vec)
        seed_id = unique_segments[seed].item()
        is_seed_included = np.array([seed_id in inst for inst in curr_instances])
        return curr_instances[is_seed_included][0]
    elif mode == "avg":  # return segment with highest average eigenvector value
        unique_segments = unique_segments.cpu().numpy()

        # get vector averages in partition
        avg_mean_values = []
        for sep in curr_instances:
            avg_mean_values += [
                np.mean(second_smallest_vec[np.isin(unique_segments, list(sep))])
            ]

        max_avg_item = np.argmax(avg_mean_values)
        return curr_instances[max_avg_item]

    elif mode == "largest":  # return largest segment by segment number
        segment_sizes = np.array([len(inst) for inst in curr_instances])
        max_size_item = np.argmax(segment_sizes)
        return curr_instances[max_size_item]

    elif mode == "all":  # return all segments
        return set(curr_mask_segment_ids)
    else:
        raise NotImplementedError


def segment_ids_to_mask(selected_ids, unique_segments):
    selected_map = torch.zeros_like(unique_segments)
    for s_id in selected_ids:
        segment_index = (unique_segments == s_id).nonzero(as_tuple=True)[0]
        selected_map[segment_index] = 1

    return selected_map.bool()


def segments_to_mask(selected_segments, unique_segments):
    selected_map = torch.zeros_like(unique_segments)
    for s_id in selected_segments:
        segment_index = (unique_segments == s_id).nonzero(as_tuple=True)[0]
        selected_map[segment_index] = 1

    return selected_map.bool()


def assign_openscene_feats(scene_name, coords, dataset, config):
    # Load full res data
    scene_index = dataset.scene_names.index(scene_name)
    full_res_coords, *_ = dataset.load_scene_data(scene_index)

    # Load full res features
    base_path = config.misc.preprocessed_path
    vertex_features = torch.zeros((full_res_coords.shape[0], 512))

    scene_chunk_paths = glob.glob(os.path.join(base_path, f"{scene_name}_*.pt"))
    for path in scene_chunk_paths:
        data = torch.load(path)
        feat_3d, mask_chunk = data["feat"], data["mask_full"]
        vertex_features[mask_chunk] = feat_3d.squeeze().float()
    vertex_features = vertex_features.to(coords.device)

    # Associate high res vertex features to low res coords
    coord_features = torch.zeros((coords.shape[0], 512))
    scaled_full_res_coords = (full_res_coords / dataset.VOXEL_SIZE) - 0.5
    high_res_tree = KDTree(scaled_full_res_coords)  # for the voxelization shift
    _, high_res_low_res_matches = high_res_tree.query(coords[:, 1:].cpu().numpy(), k=1)
    high_res_low_res_matches = high_res_low_res_matches.flatten()
    coord_features = vertex_features[high_res_low_res_matches].detach()

    return coord_features


def encode_scene_feats(
    scene_batch,
    model,
    config,
    dataset,
    use_image_data=False,
    whiten=False,
    device="cuda",
):
    # Scene batch is used to determine the name and unpack all necessary inputs
    # The dimensions determine if we want 2D or 3D projections

    # hier kannst du alternativ viel von deinem code einsetzen, todo: patch wise feature extraction

    torch.cuda.empty_cache()
    (
        coords,
        feats,
        target,
        scene_name,
        images,
        camera_poses,
        color_intrinsics,
        segment_ids,
        seg_connectivity,
        *transform,
    ) = scene_batch
    scene_name = scene_name[0]

    scene_path = f"outputs/{scene_name}_{use_image_data}_feats.pth"
    if os.path.isfile(scene_path):
        return torch.load(scene_path)
    else:
        coords, feats, target = (
            coords.to(device),
            feats.to(device),
            target.to(device),
        )

        sinput = SparseTensor(feats, coords)

        # Initialize models and encode the features
        if use_image_data:
            feature_projecter = Project2DFeaturesCUDA(
                width=1280,#dataset.depth_shape[1],
                height=960,#dataset.depth_shape[0],
                voxel_size=dataset.VOXEL_SIZE,
                config=config,
            )

            images, camera_poses, color_intrinsics = (
                images.to(device),
                camera_poses.to(device),
                color_intrinsics.to(device),
            )
            scene_key_feats = torch.zeros(
                (sinput.C.shape[0], 607), device=device
            )
            scene_query_feats = torch.zeros(
                (sinput.C.shape[0], 607), device=device
            )
            #images = images.reshape(1,3,968,1296)
            images = torch.nn.functional.interpolate(images,(960,1280))
            camera_poses = camera_poses.unsqueeze(0)
            if config.image_data.load_precomputed:
                return assign_openscene_feats(scene_name, coords, dataset, config)
            else:
                # Project all image features to the voxel grid
                camera_poses[0,0,:3, :] /= 0.02  # adapt to voxel coord system
                
                
                for i, img in enumerate(images):
                    # insert patch wise feature extraction here
                    with torch.no_grad():
                        key_features = model(
                            img
                        )  # (batch, img_num, c, h, w)
                        key_features = key_features.permute(0, 2, 3, 1).unsqueeze(0).unsqueeze(0)
                        projected_key_features, mapping2dto3d_num = feature_projecter(
                            encoded_2d_features=key_features[0],
                            coords=sinput.C,
                            view_matrix=camera_poses[0, i].unsqueeze(0).unsqueeze(0),
                            intrinsic_params=color_intrinsics[0],#/2,
                        )
                        print(projected_key_features.sum())
                    projected_points = mapping2dto3d_num > 0
                    scene_key_feats[projected_points] = torch.stack(
                        (
                            scene_key_feats[projected_points],
                            projected_key_features[projected_points],
                        )
                    ).mean(0)
                print(scene_key_feats.sum())

                return scene_key_feats

        else: # todo implement this for csc feature extraction
            # Feed forward on pretrained model
            _, feature_maps = model(sinput)
            encoded_feats = feature_maps[f"res_{config.freemask.resolution_scale}"]
            lr_coords = encoded_feats.C[:, 1:].cpu().numpy()

            # Associate low res features to
            from scipy.spatial import KDTree

            low_res_tree = KDTree(lr_coords)
            _, low_res_high_res_matches = low_res_tree.query(
                coords[:, 1:].cpu().numpy(), k=1
            )
            low_res_high_res_matches = low_res_high_res_matches.flatten()

            encoded_feats = encoded_feats.F[low_res_high_res_matches].detach()

            return whiten_feats(encoded_feats) if whiten else encoded_feats


def aggregate_features(encoded_features, segment_ids, seg_connectivity, config):
    device = encoded_features.device

    # get unique IDs at low res
    unique_segments = segment_ids.unique()

    # Get queries by averaging over the segment point feats
    segment_feats = torch.zeros((len(unique_segments), encoded_features.shape[1]))
    print(encoded_features.sum())
    # Only average valid features for every segments
    valid_mask = torch.any(encoded_features != 0, dim=-1)
    for i, s_id in enumerate(unique_segments):
        segment_mask = valid_mask * (segment_ids == s_id)
        if segment_mask.sum() > 0:
            valid_segment_feats = encoded_features[valid_mask * (segment_ids == s_id)]
            segment_feats[i, :] = (
                valid_segment_feats.max(0)[0]
                if config.freemask.aggregation_mode == "max"
                else valid_segment_feats.mean(0)
            )
        else:
            segment_feats[i, :] = 0.0
    print(torch.unique(valid_mask, return_counts=True))
    # precompute connectivity dict
    connectivity_dict = {}
    for s_id in unique_segments:
        connectivity_dict[s_id.item()] = set(
            (seg_connectivity[seg_connectivity[:, 0] == s_id, 1].cpu().numpy())
        )

    # Finally get aggregated features from segment average
    aggregated_features = segment_feats.clone().detach()

    # Replace zero features with the mean of the connected components
    # Get segments with zero values
    zero_segments = unique_segments[torch.all(aggregated_features == 0, dim=-1)]

    for zero_segment in zero_segments:
        segment_index = (unique_segments == zero_segment).nonzero(as_tuple=True)[0]

        # Get connected segments,
        # filter out the ones which are zeros themselves
        connected_zero_segments = seg_connectivity[
            seg_connectivity[:, 0] == zero_segments[0]
        ][:, 1]
        connected_zero_segment_indices = torch.LongTensor(
            [
                (unique_segments == s_id).nonzero(as_tuple=True)[0]
                for s_id in connected_zero_segments
            ]
        )
        connected_zero_segment_feats = aggregated_features[
            connected_zero_segment_indices
        ]
        connected_zero_segment_feats = connected_zero_segment_feats[
            torch.any(connected_zero_segment_feats != 0.0, dim=-1)
        ]

        # Orig zero segment should be mean of valid connected components
        if len(connected_zero_segment_feats) != 0:
            new_segment_feature = torch.mean(connected_zero_segment_feats, dim=0)
        else:
            # Simply mean feature accross scene
            new_segment_feature = torch.mean(aggregated_features, dim=0)

        aggregated_features[segment_index] = new_segment_feature

    aggregated_features = aggregated_features.to(device)
    return aggregated_features, unique_segments


def maskcut3d(
    aggregated_features,
    unique_segments,
    seg_connectivity,
    segment_ids,
    scene_coords,
    scene_colors,
    affinity_tau=0.65,
    max_number_of_instances=20,
    similarity_metric="cos",
    max_extent_ratio=0.8,
    max_surface_ratio=0.3,
    eps=1e-5,
    min_segment_size=4,
    separation_mode="max",
):
    bipartitions = []
    foreground_segments = set()
    device = scene_coords.device

    if len(unique_segments) < 3:
        return np.ones(len(unique_segments), dtype=np.bool).reshape(1, -1)

    num_segments = len(unique_segments)
    for i in range(max_number_of_instances):
        if i == 0:
            painting = torch.zeros(num_segments, device=device)
        else:
            aggregated_features, painting = get_masked_affinity_matrix(
                painting, aggregated_features, current_mask
            )

            # construct the affinity matrix
        A, D = get_affinity_matrix(
            aggregated_features,
            tau=affinity_tau,
            eps=eps,
            normalize_sim=True,
            similarity_metric=similarity_metric,
        )
        A[painting.cpu().bool()] = eps
        A[:, painting.cpu().bool()] = eps

        # get the second-smallest eigenvector
        try:
            _, second_smallest_vec = second_smallest_eigenvector(A, D)
        except ValueError:
            debug = 0
        # _, second_smallest_vec = second_smallest_eigenvector(A, D)

        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)
        #print(np.unique(bipartition,return_counts=True))

        # Get foreground points
        fg_coords = np.zeros((0, 3))
        fg_colors = np.zeros((0, 3))

        # visualize bipartition with segments
        for fg_segment in unique_segments[bipartition]:
            segment_mask = segment_ids == fg_segment
            fg_coords = np.append(
                fg_coords, scene_coords[segment_mask].cpu().numpy(), axis=0
            )
            fg_colors = np.append(
                fg_colors, scene_colors[segment_mask].cpu().numpy(), axis=0
            )

        # Calculate scene extents to differentiate btw fg and bg
        scene_mins, scene_maxes = scene_coords.cpu().numpy().min(
            0
        ), scene_coords.cpu().numpy().max(0)
        partition_mins, partition_maxes = fg_coords.min(0), fg_coords.max(0)
        scene_extents, partition_extents = (
            scene_maxes - scene_mins,
            partition_maxes - partition_mins,
        )

        # Flip partition if extent is too large and more than half is accepted as a foreground
        is_scene_extent_condition = np.all(
            partition_extents[:2] > scene_extents[:2] * max_extent_ratio
        )
        is_fg_ratio_condition = bipartition.sum() / len(bipartition) > max_extent_ratio
        if is_fg_ratio_condition:
            bipartition = np.logical_not(bipartition)
            second_smallest_vec = second_smallest_vec * -1
            # print(f'Bipartition reversed')
        seed = np.argmax(second_smallest_vec)

        # Do the bipartition separation
        separated_seed_partition = separate_segments(
            bipartition,
            second_smallest_vec,
            unique_segments,
            seg_connectivity,
            mode=separation_mode,
        )
        separated_seed_pseudo_mask = segment_ids_to_mask(
            separated_seed_partition, unique_segments
        )

        # Check IoU with previous foregrounds
        IoU = len(
            set.intersection(separated_seed_partition, foreground_segments)
        ) / len(separated_seed_partition)
        if IoU > 0.5:
            #print(f'Skipped current mask with high IoU score of {IoU}')
            current_mask = separated_seed_pseudo_mask
            continue
        if len(separated_seed_partition) < min_segment_size:
            #print(f'Skipped current mask with too small size {len(separated_seed_partition)}')
            current_mask = separated_seed_pseudo_mask
            continue

        # Visualize with the updated bipartitions
        fg_coords_bp, fg_coords_seed = np.zeros((0, 3)), np.zeros((0, 3))
        fg_colors_bp, fg_colors_seed = np.zeros((0, 3)), np.zeros((0, 3))
        for fg_segment in unique_segments[bipartition].cpu().numpy():
            segment_mask = segment_ids == fg_segment
            fg_coords_bp = np.append(
                fg_coords_bp, scene_coords[segment_mask].cpu().numpy(), axis=0
            )
            fg_colors_bp = np.append(
                fg_colors_bp, scene_colors[segment_mask].cpu().numpy(), axis=0
            )

            # Add the segment to the pseudo mask if seed was included
            if fg_segment in separated_seed_partition:
                fg_coords_seed = np.append(
                    fg_coords_seed, scene_coords[segment_mask].cpu().numpy(), axis=0
                )
                fg_colors_seed = np.append(
                    fg_colors_seed, scene_colors[segment_mask].cpu().numpy(), axis=0
                )

        # Mask out the parts which has been already accepted as foreground
        separated_seed_partition_masked = separated_seed_partition - foreground_segments
        bipartitions += [
            segment_ids_to_mask(separated_seed_partition_masked, unique_segments)
            .cpu()
            .numpy()
        ]
        foreground_segments = foreground_segments.union(separated_seed_partition)
        # print(f'Painted out {len(separated_seed_partition)} segments')

        # Finally save the segment mask as current mask for nex iteration
        current_mask = separated_seed_pseudo_mask

    # In the end take all reamining background, separate them a

    bipartitions = (
        np.stack(bipartitions)
        if len(bipartitions) > 0
        else np.zeros((0, len(segment_ids)))
    )
    return bipartitions


def segment_scene(
    coords,
    feats_2d,
    feats_3d,
    config,
    segment_ids,
    seg_connectivity,
    scene_colors
):
    
    # Aggregate features
    aggregated_feats_2d, unique_segments = aggregate_features(
            feats_2d, segment_ids, seg_connectivity, config
    )
    aggregated_feats_3d, unique_segments = aggregate_features(
            feats_3d, segment_ids, seg_connectivity, config
    )
    aggregated_features = (aggregated_feats_2d,aggregated_feats_3d)
    # Start the iterative NCut algorithm
    bipartitions = maskcut3d(
        aggregated_features,
        unique_segments,
        seg_connectivity,
        segment_ids,
        coords,
        scene_colors
        # affinity_tau=config.freemask.affinity_tau,
        # max_number_of_instances=max_number_of_instances,
        # similarity_metric=config.freemask.similarity_metric,
        # min_segment_size=config.freemask.min_segment_size,
        # separation_mode=config.freemask.separation_mode,
        # max_extent_ratio=config.freemask.max_extent_ratio,
    )

    return bipartitions, aggregated_features
import json
@hydra.main(config_path="../conf", config_name="config_base_instance_segmentation.yaml")
def main(config):
    device = "cuda"

    mask_save_base = "/mnt/hdd/ncut_masks/"

    # Start main iteration over the dataset
    for scene_name in os.listdir("/mnt/hdd/ncut_features/2d/lseg/"):
    
        coords_2d = np.load("/mnt/hdd/ncut_features/2d/lseg/" + scene_name + "/coords.npy")
        feats_2d = np.load("/mnt/hdd/ncut_features/2d/lseg/" + scene_name + "/lseg_feats.npy")
        coords_3d = np.load("/mnt/hdd/ncut_features/3d/csc/" + scene_name + "/coords.npy")
        feats_3d = np.load("/mnt/hdd/ncut_features/3d/csc/" + scene_name + "/csc_feats.npy")
        result = np.array_equal(coords_2d, coords_3d)
        assert(result==True,"coords_2d and coords_3d not equal")

        f = open("/mnt/hdd/scannet_test_segments/" + scene_name + "_vh_clean_2.0.005000.segs.json")
        data = json.load(f)
        segment_ids = np.array(data["segIndices"],dtype=np.int32)
        seg_connectivity = np.array(data["segConnectivity"],dtype=np.int32)
        f.close
        
        # Instantiate PCA and fit-transform the data
        pca = PCA(n_components=3)
        feats = pca.fit_transform(feats_2d)

        coords = torch.tensor(coords_2d,device=device)
        feats_2d = torch.tensor(feats_2d,device=device)
        feats_3d = torch.tensor(feats_3d,device=device)
        segment_ids = torch.tensor(segment_ids,device=device)
        seg_connectivity = torch.tensor(seg_connectivity,device=device)   
        feats = torch.tensor(feats,device=device)   


        # Create output dir for images
        if not os.path.exists(mask_save_base):
            os.makedirs(mask_save_base)

        #Skip if already processed
        if os.path.exists(f"{mask_save_base}/{scene_name[0]}_cloud.npy"):
            print(f"Scene already processed: {scene_name[0]}")
            # print(f'Scene already processed: {scene_name[0]}, saving cloud data only')
            #
            # # Here we have to save the masks with all info instead of only points
            # full_res_scene_data = dataset.get_full_cloud_by_scene_name(scene_name[0])
            # full_res_coords, full_res_feats, full_res_labels, full_res_instance_ids, *_ = full_res_scene_data
            #
            # # Associate order of points
            # small_res_tree = KDTree(coords[:, 1:].cpu().numpy() + 0.5)  # for rounding shift at 2cm resolution the rounding shift was 1 voxel size
            # _, lr_hr_matches = small_res_tree.query(full_res_coords / dataset.VOXEL_SIZE, k=1)
            #
            # full_res_segment_ids = segment_ids.cpu().numpy()[lr_hr_matches]
            #
            # all_cloud_save = np.concatenate((full_res_coords, full_res_feats, full_res_labels[:, None], full_res_instance_ids[:, None], full_res_segment_ids[:, None]), axis=1).astype(np.single)
            # np.save(f'{freemask_save_base}/{scene_name[0]}_cloud.npy', all_cloud_save)
        #     continue

        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        orig_pcd.colors = o3d.utility.Vector3dVector(feats.cpu().numpy() + 0.5)
        orig_pcd.estimate_normals()

        segment_colors = (
            torch.stack(
                (
                    segment_ids * 217 % 256,
                    segment_ids * 217 % 311,
                    segment_ids * 217 % 541,
                )
            ).T
            % 256
        )
        segment_pcd = o3d.geometry.PointCloud()
        segment_pcd.points = o3d.utility.Vector3dVector(
            coords.cpu().numpy()
        )
        segment_pcd.colors = o3d.utility.Vector3dVector(
            segment_colors.cpu().numpy() / 255.0
        )

        orig_coords = coords.cpu().numpy() * 0.02
        orig_colors = (feats.cpu().numpy() + 0.5) * 255.0
        orig_normals = np.array(orig_pcd.normals)

        # Run the iterative cut and mask algorithm
        bipartitions, aggregated_features = segment_scene(
            coords,
            feats_2d,
            feats_3d,
            config,
            segment_ids,
            seg_connectivity,
            feats
        )
    

        # Calculate inverse segment mapping
        unique_segments = segment_ids.unique()
        inverse_segment_mapping = torch.zeros_like(segment_ids)
        for i, s_id in enumerate(unique_segments):
            segment_mask = segment_ids == s_id
            inverse_segment_mapping[segment_mask] = i
        inverse_segment_mapping = inverse_segment_mapping.cpu().numpy()

        # Update bipartitions to be on point level instead of segment level
        bipartitions = bipartitions.T[inverse_segment_mapping]
        num_instances = bipartitions.shape[1]

        # Visualize the final bipartitions
        if config.test.visualize:
            instance_colors = np.zeros(coords.shape)
            for i in reversed(range(num_instances)):
                instance_colors[bipartitions[:, i]] = np.random.rand(3)

            instance_pcd = o3d.geometry.PointCloud()
            instance_pcd.points = o3d.utility.Vector3dVector(
                coords.cpu().numpy()
            )
            instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)

            v = vis.Visualizer()
            v.add_points(
                "RGB Input",
                orig_coords,
                colors=orig_colors,
                normals=orig_normals,
                visible=True,
                point_size=50,
            )

            v.add_points(
                "Segments",
                orig_coords,
                colors=segment_colors.cpu().numpy(),
                normals=orig_normals,
                visible=True,
                point_size=50,
            )

            if isinstance(aggregated_features, tuple):
                aggregated_features = aggregated_features[0]

            pca = PCA(n_components=3)
            projected_backbone_feats = pca.fit_transform(
                aggregated_features.cpu().numpy()
            )
            projected_backbone_feats = (
                projected_backbone_feats - projected_backbone_feats.min()
            )
            projected_backbone_feats = (
                projected_backbone_feats / projected_backbone_feats.max()
            )

            _, inverse_map = np.unique(segment_ids.cpu().numpy(), return_inverse=True)
            v.add_points(
                "PCA",
                orig_coords,
                colors=projected_backbone_feats[inverse_map] * 255.0,
                normals=orig_normals,
                visible=True,
                point_size=50,
            )

            v.add_points(
                "Instances",
                orig_coords,
                colors=instance_colors * 255.0,
                normals=orig_normals,
                visible=True,
                point_size=50,
            )

            #v.save(f"{freemask_save_base}/../visualize/{scene_name[0]}")

            # o3d.visualization.draw_geometries([instance_pcd, orig_pcd])

        # Get full res scene data to save for dataset
        full_res_coords = orig_coords
        # full_res_scene_data = dataset.get_full_cloud_by_scene_name(scene_name[0])
        # (
        #     full_res_coords,
        #     full_res_feats,
        #     full_res_labels,
        #     full_res_instance_ids,
        #     *_,
        # ) = full_res_scene_data

        # Associate order of points
        small_res_tree = KDTree(
            coords.cpu().numpy() + 0.5
        )  # for rounding shift at 2cm resolution the rounding shift was 1 voxel size
        _, lr_hr_matches = small_res_tree.query(
            full_res_coords / 0.02, k=1
        )

        full_res_segment_ids = segment_ids.cpu().numpy()[lr_hr_matches]
        all_masks_save = bipartitions[lr_hr_matches]

        # Save Felzenswalb masks only instead of bipartitions if specified
        if "felzenswalb_partitions" in config:
            num_segments = len(np.unique(full_res_segment_ids))
            all_masks_save = np.zeros(
                (all_masks_save.shape[0], num_segments), dtype=bipartitions.dtype
            )
            for i, seg_id in enumerate(np.unique(full_res_segment_ids)):
                seg_mask = full_res_segment_ids == seg_id
                all_masks_save[seg_mask, i] = True

        all_cloud_save = full_res_coords.astype(
            np.single
        )  # np.concatenate((full_res_coords, full_res_feats, full_res_labels[:, None], full_res_instance_ids[:, None], full_res_segment_ids[:, None]), axis=1).astype(np.single)
        np.save(f"{mask_save_base}/{scene_name[0]}_cloud.npy", all_cloud_save)
        np.save(f"{mask_save_base}/{scene_name[0]}_masks.npy", all_masks_save)


if __name__ == "__main__":
    main()
