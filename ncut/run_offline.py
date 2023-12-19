import torch
import hydra
import sys
import numpy as np
import torch.nn.functional as F
import os
from scipy.linalg import eigh
import logging


log = logging.getLogger(__name__)


def normalize_mat(A, eps=1e-5):
    A -= np.min(A[np.nonzero(A)]) if np.any(A > 0) else 0
    A[A < 0] = 0.0
    A /= A.max() + eps
    return A


def get_affinity_matrix(feats, tau=0.15, eps=1e-5, normalize_sim=True):
    """
    create unweighted adjacency matrix, edge if mean of feature similarity above tau
    """
    # get affinity matrix via measuring patch-wise cosine similarity

    if not isinstance(feats, tuple):  # single modality feature
        feats_a = F.normalize(feats, p=2, dim=-1)
        A = feats_a @ feats_a.T

        # normalize attentions if requested
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


def segment_ids_to_mask(selected_ids, unique_segments):
    selected_map = torch.zeros_like(unique_segments)
    for s_id in selected_ids:
        segment_index = (unique_segments == s_id).nonzero(as_tuple=True)[0]
        selected_map[segment_index] = 1

    return selected_map.bool()


def aggregate_features(encoded_features, segment_ids, seg_connectivity, config):
    """
    compute mean features over felzenszwalb segments (geometric oversegmentation)
    """
    device = encoded_features.device

    # get unique IDs at low res
    unique_segments = segment_ids.unique()

    # Get queries by averaging over the segment point feats
    segment_feats = torch.zeros((len(unique_segments), encoded_features.shape[1]))

    # Only average valid features for every segments
    valid_mask = torch.any(encoded_features != 0, dim=-1)
    for i, s_id in enumerate(unique_segments):
        segment_mask = valid_mask * (segment_ids == s_id)
        if segment_mask.sum() > 0:
            valid_segment_feats = encoded_features[valid_mask * (segment_ids == s_id)]
            segment_feats[i, :] = valid_segment_feats.mean(0)
        else:
            segment_feats[i, :] = 0.0

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


def maskcut3d(
    aggregated_features,
    unique_segments,
    seg_connectivity,
    segment_ids,
    scene_coords,
    # scene_colors,
    affinity_tau=0.65,
    max_number_of_instances=20,
    # similarity_metric="cos",
    max_extent_ratio=0.8,
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

        # Get foreground points
        # fg_coords = np.zeros((0, 3))
        # fg_colors = np.zeros((0, 3))

        # visualize bipartition with segments
        # for fg_segment in unique_segments[bipartition]:
        #     segment_mask = segment_ids == fg_segment
        #     fg_coords = np.append(
        #         fg_coords, scene_coords[segment_mask].cpu().numpy(), axis=0
        #     )
        #     fg_colors = np.append(
        #         fg_colors, scene_colors[segment_mask].cpu().numpy(), axis=0
        #     )

        # Calculate scene extents to differentiate btw fg and bg
        # scene_mins, scene_maxes = scene_coords.cpu().numpy().min(
        #     0
        # ), scene_coords.cpu().numpy().max(0)
        # partition_mins, partition_maxes = fg_coords.min(0), fg_coords.max(0)
        # scene_extents, partition_extents = (
        #     scene_maxes - scene_mins,
        #     partition_maxes - partition_mins,
        # )

        # Flip partition if extent is too large and more than half is accepted as a foreground
        # is_scene_extent_condition = np.all(
        #     partition_extents[:2] > scene_extents[:2] * max_extent_ratio
        # )
        is_fg_ratio_condition = bipartition.sum() / len(bipartition) > max_extent_ratio
        if is_fg_ratio_condition:
            bipartition = np.logical_not(bipartition)
            second_smallest_vec = second_smallest_vec * -1
            # print(f'Bipartition reversed')
        # seed = np.argmax(second_smallest_vec)

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
            # print(f'Skipped current mask with high IoU score of {IoU}')
            current_mask = separated_seed_pseudo_mask
            continue
        if len(separated_seed_partition) < min_segment_size:
            # print(f'Skipped current mask with too small size {len(separated_seed_partition)}')
            current_mask = separated_seed_pseudo_mask
            continue

        # Visualize with the updated bipartitions
        # fg_coords_bp, fg_coords_seed = np.zeros((0, 3)), np.zeros((0, 3))
        # fg_colors_bp, fg_colors_seed = np.zeros((0, 3)), np.zeros((0, 3))
        # for fg_segment in unique_segments[bipartition].cpu().numpy():
        #     segment_mask = segment_ids == fg_segment
        #     fg_coords_bp = np.append(
        #         fg_coords_bp, scene_coords[segment_mask].cpu().numpy(), axis=0
        #     )
        #     fg_colors_bp = np.append(
        #         fg_colors_bp, scene_colors[segment_mask].cpu().numpy(), axis=0
        #     )

        #     # Add the segment to the pseudo mask if seed was included
        #     if fg_segment in separated_seed_partition:
        #         fg_coords_seed = np.append(
        #             fg_coords_seed, scene_coords[segment_mask].cpu().numpy(), axis=0
        #         )
        #         fg_colors_seed = np.append(
        #             fg_colors_seed, scene_colors[segment_mask].cpu().numpy(), axis=0
        #         )

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
    feats_3d,
    feats_2d,
    config,
    segment_ids,
    seg_connectivity,
):
    """
    applies geometric oversegmentation, applies normalized cut
    """
    # geometric oversegmentation
    aggregated_3d_features, unique_segments = aggregate_features(
        feats_3d, segment_ids, seg_connectivity, config
    )
    aggregated_2d_features, unique_segments = aggregate_features(
        feats_2d, segment_ids, seg_connectivity, config
    )
    aggregated_features = (aggregated_3d_features, aggregated_2d_features)

    # Start the iterative NCut algorithm
    bipartitions = maskcut3d(
        aggregated_features,
        unique_segments,
        seg_connectivity,
        segment_ids,
        coords,
        # feats,
        affinity_tau=config.ncut.affinity_tau,
        max_number_of_instances=config.ncut.max_number_of_instances,
        min_segment_size=config.ncut.min_segment_size,
        separation_mode=config.ncut.separation_mode,
        max_extent_ratio=config.ncut.max_extent_ratio,
    )

    return bipartitions, aggregated_features


def mean_instance_feature(segment_featues, segmentwise_instances):
    """
    arguments:
    segment_features: np(n_segments, dim_feature)
    segmentwise_instances: np(n_segments, n_instances) (one hot instances)

    returns:
    mean segment features np(n_instances, dim_feat)
    segments that do not belong to any instance or have sum(feature) == 0 are ignored
    """
    n_instances = segmentwise_instances.shape[1]
    dim_feat = segment_featues.shape[1]
    aggr_labels = np.zeros(shape=(n_instances, dim_feat))
    counts = np.zeros(shape=(n_instances, 1))
    for i, s_feat in enumerate(segment_featues):
        if segmentwise_instances[i].sum() == 0:
            continue  # skip segments that do not belong to any instance
        if s_feat.sum() == 0:
            continue  # skip segments in case none of its point was hit by a feature projection
        instance = np.argmax(segmentwise_instances[i])
        aggr_labels[instance] += s_feat
        counts[instance] += 1
    return aggr_labels / counts


@hydra.main(config_path="../conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg):
    sys.path.append(hydra.utils.get_original_cwd())

    from ncut.ncut_datasets import NormalizedCutDataset
    from ncut.visualize import (
        visualize_segments,
        visualize_instances,
        visualize_3d_feats,
    )

    device = torch.device("cuda:0")
    loader = NormalizedCutDataset.dataloader_from_hydra(cfg.ncut.data, only_first=True)

    n_scenes = len(loader)
    for i_scene, sample in enumerate(loader):
        log.info(f"** {sample['scene_name']} ({i_scene + 1}/{n_scenes})")

        coords = sample["coords"][0].to(device)  # tens(n_points, 3)
        feats_3d = sample["feats_3d"][0].to(device)  # tens(n_points, dim_feat_3d)
        feats_2d = sample["feats_2d"][0].to(device)  # tens(n_points, dim_feat_2d)
        segment_ids = sample["segment_ids"][0].to(device)  # tens(n_points,)
        segment_connectivity = sample["segment_connectivity"][0].to(
            device
        )  # tens(-1, 2)
        scene_name = sample["scene_name"][0]

        # segment scene using normalized cut
        (
            bipartitions,  # np(n_segments, n_instances) col-wise one hot representation of instance
            (
                segment_feats_3d,  # tens(n_segments, dim_feat_3d)
                segment_feats_2d,  # tens(n_segments, dim_feat_2d)
            ),
        ) = segment_scene(
            coords,
            feats_3d,
            feats_2d,
            cfg,
            segment_ids,
            segment_connectivity,
        )

        # Calculate inverse segment mapping
        unique_segments = segment_ids.unique()
        segment_indices = torch.zeros_like(segment_ids)
        for i, segment_id in enumerate(unique_segments):
            segment_mask = segment_ids == segment_id
            segment_indices[segment_mask] = i
        segment_indices = segment_indices.cpu().numpy()

        # Update bipartitions to be on point level instead of segment level
        pointwise_instances = bipartitions.T[segment_indices].astype(
            int
        )  # np(n_points, n_instances) row-wise one hot representation of instances

        # get labels by taking mean 2d feature over instances
        labels = mean_instance_feature(
            segment_featues=segment_feats_2d.cpu().numpy(),
            segmentwise_instances=bipartitions.T,
        )  # np(n_instances, dim_feat_2d)

        # save everything as np arrays
        scan_dir = os.path.join(cfg.ncut.save_dir, scene_name)
        os.makedirs(scan_dir, exist_ok=True)
        coords_file = os.path.join(scan_dir, cfg.ncut.coords_filename)
        instances_file = os.path.join(scan_dir, cfg.ncut.instances_filename)
        labels_file = os.path.join(scan_dir, cfg.ncut.labels_filename)
        np.save(coords_file, coords.cpu().numpy())  # (n_points, 3)
        log.info(f"Saved: {coords_file}")
        np.save(instances_file, pointwise_instances)  # (n_points, n_instances) (onehot)
        log.info(f"Saved: {instances_file}")
        np.save(labels_file, labels)  # (n_instances, feature_dim_2d)
        log.info(f"Saved: {labels_file}")

        # visualize geometric oversegmentation and instances
        segment_visualization_filename = cfg.ncut.get(
            "segment_visualization_filename", None
        )
        if segment_visualization_filename:
            segment_visualization_file = os.path.join(
                scan_dir, segment_visualization_filename
            )
            visualize_segments(
                coords.cpu().numpy(),
                unique_segments.cpu().numpy(),
                segment_ids,
                segment_visualization_file,
            )
            log.info(f"Saved: {segment_visualization_file}")
        instance_visualization_filename = cfg.ncut.get(
            "instance_visualization_filename", None
        )
        if instance_visualization_filename:
            instance_visualization_file = os.path.join(
                scan_dir, instance_visualization_filename
            )
            visualize_instances(
                coords.cpu().numpy(),
                pointwise_instances,
                instance_visualization_file,
            )
            log.info(f"Saved: {instance_visualization_file}")


if __name__ == "__main__":
    main()
