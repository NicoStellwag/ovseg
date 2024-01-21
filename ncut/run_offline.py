from sklearn.decomposition import PCA
import torch
import hydra
import sys
import numpy as np
import torch.nn.functional as F
import os
from scipy.linalg import eigh
from sklearn.manifold import spectral_embedding
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.elbow import elbow
from sklearn.cluster import k_means
from shutil import rmtree

import logging


log = logging.getLogger(__name__)


def normalize_mat(A, eps=1e-5):
    A -= np.min(A[np.nonzero(A)]) if np.any(A > 0) else 0
    A[A < 0] = 0.0
    A /= A.max() + eps
    return A


def get_affinity_matrix(
    feats, tau=0.15, eps=1e-5, normalize_sim=True, dim_reduce_2d=None, binarize=True
):
    """
    create unweighted adjacency matrix, edge if mean of feature similarity above tau
    """
    # dim reduce if specified
    if dim_reduce_2d:
        n_samples = (
            feats.shape[0] if not isinstance(feats, tuple) else feats[1].shape[0]
        )
        pca = PCA(n_components=min(dim_reduce_2d, n_samples))
        if not isinstance(feats, tuple):
            dev = feats.device
            feats = torch.from_numpy(pca.fit_transform(feats.cpu().numpy())).to(dev)
        else:
            dev = feats[1].device
            reduced_2d = torch.from_numpy(pca.fit_transform(feats[1].cpu().numpy())).to(
                dev
            )
            feats = (feats[0], reduced_2d)

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

    if binarize:
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
    dim_reduce_2d=None,
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
            dim_reduce_2d=dim_reduce_2d,
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


# spectral clustering implementation
# =================


def feature_spectral_embeddings(
    feats, unique_segments, seg_connectivity, dim_reduce_2d=None
):
    """
    arguments:
    feats: tens(n_segs, feature_dim) or tuple(tens(n_segs, feat_dim1), tens(n_segs, feat_dim2))
    unique_segments: np(n_segs,): unique segment ids in the same order of features
    seg_connectivity: neighborhood graph of segments tens(n_edges, 2)

    returns: tuple(U, A)
    U: row-wise spectral embeddings np(n_seg, n_seg)
    A: affinity matrix np(n_seg, n_seg)

    (if feats is tuple the affinity is mean of both affinity matrices)
    """
    A, D = get_affinity_matrix(feats, binarize=False, dim_reduce_2d=dim_reduce_2d)

    # only neighboring segments are connected
    segment_id_to_index = {}
    for s_index, s_id in enumerate(unique_segments):
        segment_id_to_index[s_id.item()] = s_index
    mask = np.zeros_like(A)
    for edge in seg_connectivity:
        u, v = edge[0].item(), edge[1].item()
        i_u, i_v = segment_id_to_index[u], segment_id_to_index[v]
        mask[i_u, i_v] = 1.0
        mask[i_v, i_u] = 1.0
    A = A * mask

    # spectral embeddings (eigenvectors of laplacian)
    U = spectral_embedding(
        A,
        norm_laplacian=True,  # (normed L: normalized cut, unnormed L: ratio cut)
        drop_first=False,  # first eigenvec is const for fully connected graph, we have missing edges though
    )

    return U, A


def filter_low_confidence_instances(U, dense_bipartitions):
    """
    confidence means low intra cluster embedding variance

    arguments:
    U: row-wise spectral embeddings of graph np(n_segments, dim_emb)
    dense_bipartitions: np(n_instances, n_segments)

    returns:
    filtered_bipartitions: np(n_instances_filtered, n_segments)
    """
    # compute intra cluster embedding variances
    n_clusters = dense_bipartitions.shape[0]
    clusters = dense_bipartitions.argmax(axis=0)
    variances = np.empty(shape=(n_clusters))
    for cluster_id in range(n_clusters):
        cluster_embeddings = U[clusters == cluster_id]
        cluster_centroid = cluster_embeddings.mean(axis=0)
        cluster_variance = np.mean(
            np.sum((cluster_embeddings - cluster_centroid) ** 2, axis=-1)
        )
        variances[cluster_id] = cluster_variance

    # threshold them at max delta to predecessor when sorted
    # sorted_variances = np.sort(variances)
    # var_deltas = [
    #     sorted_variances[i] - sorted_variances[i - 1]
    #     for i in range(1, len(sorted_variances))
    # ]
    # threshold_var = sorted_variances[var_deltas.index(max(var_deltas))]
    # throw_away = variances > threshold_var
    filter = variances > np.median(variances)

    # filter high variance instances
    filtered_bipartitions = dense_bipartitions.copy()
    filtered_bipartitions[filter] = 0
    return filtered_bipartitions


def spectral_cluster_3d(
    cluster_method,
    feats,
    unique_segments,
    seg_connectivity,
    max_instances,
    filter_instances,
    dim_reduce_2d,
    min_segment_size=None,
    gmeans_tolerance=None,
    xmeans_tolerance=None,
    kmeans_k=None,
):
    """
    arguments:
    feats: tens(n_segs, feature_dim) or tuple(tens(n_segs, feat_dim1), tens(n_segs, feat_dim2))
    unique_segments: np(n_segs,): unique segment ids in the same order of features
    seg_connectivity: neighborhood graph of segments tens(n_edges, 2)
    max_instances: maximum number of instances
    cluster_tolerance: max value of change of center after which clustering stops

    returns:
    instances np(n_instances, n_segments) col-wise one hot representation of instance
    """
    assert cluster_method in ["kmeans", "gmeans", "xmeans", "elbow"]

    U, A = feature_spectral_embeddings(
        feats, unique_segments, seg_connectivity, dim_reduce_2d
    )

    # k-means
    if cluster_method == "kmeans":
        assert kmeans_k, "if cluster method is kmeans, set kmeans_k"
        _, clusters, _ = k_means(X=U, n_clusters=kmeans_k)
        n_segments = A.shape[0]
        n_instances = np.unique(clusters).shape[0]
        bipartitions = np.zeros(shape=(n_instances, n_segments))
        for seg_idx, instance in enumerate(clusters):
            bipartitions[instance, seg_idx] = 1.0

    # g-means
    if cluster_method == "gmeans":
        gmargs = {}
        if gmeans_tolerance:
            gmargs["tolerance"] = gmeans_tolerance
        gm = gmeans(
            U, repeat=10, k_max=max_instances, **gmargs
        )  # a clustering algorithm that does not need fixed k
        clusters = (
            gm.process().get_clusters()
        )  # list (len=n_clusters) of lists (len=n_segs_cluster)
        n_segments = A.shape[0]
        n_instances = len(clusters)
        bipartitions = np.zeros(shape=(n_instances, n_segments))
        for i_cluster, cluster in enumerate(clusters):
            for seg in cluster:
                bipartitions[i_cluster, seg] = 1.0

    # x-means
    if cluster_method == "xmeans":
        xmargs = {}
        if xmeans_tolerance:
            xmargs["tolerance"] = xmeans_tolerance
        xm = xmeans(
            U,
            k_max=max_instances,
            criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
            **xmargs,
        )  # a clustering algorithm that does not need fixed k
        clusters = (
            xm.process().get_clusters()
        )  # list (len=n_clusters) of lists (len=n_segs_cluster)
        n_segments = A.shape[0]
        n_instances = len(clusters)
        bipartitions = np.zeros(shape=(n_instances, n_segments))
        for i_cluster, cluster in enumerate(clusters):
            for seg in cluster:
                bipartitions[i_cluster, seg] = 1.0

    # elbow
    if cluster_method == "elbow":
        el = elbow(
            U,
            kmin=1,
            kmax=max_instances,
        )  # heuristic for optimal k
        k = el.process().get_amount()
        _, clusters, _ = k_means(X=U, n_clusters=k)
        n_segments = A.shape[0]
        n_instances = np.unique(clusters).shape[0]
        bipartitions = np.zeros(shape=(n_instances, n_segments))
        for seg_idx, instance in enumerate(clusters):
            bipartitions[instance, seg_idx] = 1.0

    # filter out too small instances
    if min_segment_size and min_segment_size > 1:
        n_cluster_segments = bipartitions.sum(-1)
        filter = n_cluster_segments < min_segment_size
        bipartitions[filter] = 0

    # filter out low confidence instances
    already_filtered = bipartitions.sum(-1) == 0
    if filter_instances:
        bipartitions[~already_filtered] = filter_low_confidence_instances(
            U, bipartitions[~already_filtered]
        )

    return bipartitions


# unused
def segment_kmeans_aggregate(point_feats, segments_ids, k):
    """
    aggregates features over all nonzero points of a segment

    arguments:
    point_feats: tens(n_points, feat_dim)
    segment_ids: tens(n_points,)
    k: int - how many cluster means to save per segment

    returns (aggr_feats, unique_segments)
    aggr_feats: tens(n_segs, k, feat_dim)
    unique_segments: tens(n_segs,)
    """
    point_feats = F.normalize(point_feats, p=2, dim=-1)
    unique_segments = segments_ids.unique()
    n_segments = len(unique_segments)
    feat_dim = point_feats.shape[1]
    aggregated_feats = torch.zeros((n_segments, k, feat_dim))
    for i, s in enumerate(unique_segments):
        print(i)
        seg_feats = point_feats[segments_ids == s]
        seg_feats = seg_feats[seg_feats.sum(dim=-1).nonzero()].squeeze(dim=1)
        if len(seg_feats) > 0:  # if no non-zero feats leave it as 0
            if (
                len(seg_feats) < k
            ):  # in case there are less than k points, just repeat them to get more
                seg_feats = seg_feats.repeat(torch.ceil(k / len(seg_feats)))
            cluster_means, _, _ = k_means(X=seg_feats.cpu().numpy(), n_clusters=k)
            aggregated_feats[i] = torch.from_numpy(cluster_means)
    return aggregated_feats.to(point_feats.device), unique_segments.to(
        segments_ids.device
    )


# =================


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
    ncut_feats = (
        aggregated_features if config.ncut.use_3d_feats else aggregated_2d_features
    )

    # Start the iterative NCut algorithm
    dim_reduce_2d = (
        None if config.ncut.dim_reduce_2d == -1 else config.ncut.dim_reduce_2d
    )
    gmeans_tolerance = (
        None
        if config.ncut.spectral_gmeans_tolerance == -1
        else config.ncut.spectral_gmeans_tolerance
    )
    xmeans_tolerance = (
        None
        if config.ncut.spectral_xmeans_tolerance == -1
        else config.ncut.spectral_xmeans_tolerance
    )
    if config.ncut.method == "iterative":
        bipartitions = maskcut3d(
            ncut_feats,
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
            dim_reduce_2d=dim_reduce_2d,
        )
    elif config.ncut.method == "spectral_clustering":
        bipartitions = spectral_cluster_3d(
            cluster_method=config.ncut.spectral_cluster_method,
            feats=ncut_feats,
            unique_segments=unique_segments,
            seg_connectivity=seg_connectivity,
            max_instances=config.ncut.max_number_of_instances,
            filter_instances=config.ncut.spectral_filter_instances,
            dim_reduce_2d=dim_reduce_2d,
            gmeans_tolerance=gmeans_tolerance,
            xmeans_tolerance=xmeans_tolerance,
            kmeans_k=config.ncut.spectral_kmeans_k,
            min_segment_size=config.ncut.min_segment_size,
        )
    else:
        raise ValueError("ncut.method must be 'iterative' or 'spectral_clustering'")

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
    nonzero_instances = np.argwhere(counts != 0)
    return aggr_labels / (counts + 1e-8), nonzero_instances[:, 0]


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
    loader = NormalizedCutDataset.dataloader_from_hydra(cfg.ncut.data, only_first=False)

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

        if not cfg.ncut.use_ds_gt_segmentation:
            # segment scene using normalized cut
            (
                bipartitions,  # np(n_instances, n_segments) col-wise one hot representation of instance
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

            # Calculate segmentwise to pointwise mapping
            unique_segments = segment_ids.unique()
            segment_indices = torch.zeros_like(segment_ids)
            for i, segment_id in enumerate(unique_segments):
                segment_mask = segment_ids == segment_id
                segment_indices[segment_mask] = i
            segment_indices = segment_indices.cpu().numpy()

            # Update bipartitions to be on point level instead of segment level
            segmentwise_instances = bipartitions.T
            pointwise_instances = segmentwise_instances[segment_indices].astype(
                int
            )  # np(n_points, n_instances) row-wise one hot representation of instances
        else:
            # aggregate features over segments
            segment_feats_2d, unique_segments = aggregate_features(
                feats_2d, segment_ids, segment_connectivity, cfg
            )

            # load ground truth instance labels and remap them
            # to a consecutive sequence starting at 0
            gt_instances = (
                sample["gt_instance_labels"][0].numpy().astype(int)
            )  # np(n_points,)
            unique_gt_instances = np.unique(gt_instances[gt_instances >= 0])
            gt_instance_to_consecutive_id = {
                l: i for i, l in enumerate(unique_gt_instances)
            }
            gt_instance_to_consecutive_id[-1] = -1
            for i in range(len(gt_instances)):
                gt_instances[i] = gt_instance_to_consecutive_id[gt_instances[i]]

            # assign labels to segments by taking the label
            # that occurs for most points (there's only a small
            # number of outlier points so this should be no problem)
            n_segments = unique_segments.shape[0]
            n_instances = np.unique(gt_instances[gt_instances != -1]).shape[0]
            segment_instance_counts = np.zeros(
                shape=(n_segments, n_instances)
            )  # np(n_segments, n_instances)
            seg_id_to_unique_seg_index = {}  # cache for some speed up
            for i, inst in enumerate(gt_instances):
                if inst == -1:
                    continue
                seg_id = segment_ids[i]
                if seg_id in seg_id_to_unique_seg_index:
                    unique_seg_idx = seg_id_to_unique_seg_index[seg_id]
                else:
                    unique_seg_idx = (unique_segments == seg_id).nonzero().item()
                    seg_id_to_unique_seg_index[seg_id] = unique_seg_idx
                segment_instance_counts[unique_seg_idx, inst] = 1
            segmentwise_instances = np.zeros_like(segment_instance_counts, dtype=int)
            segmentwise_instances[
                np.arange(n_segments), np.argmax(segment_instance_counts, axis=1)
            ] = 1

            # we also need a one hot pointwise instance representation for saving
            n_points = gt_instances.shape[0]
            pointwise_instances = np.zeros(shape=(n_points, n_instances), dtype=int)
            pointwise_instances[np.arange(n_points), gt_instances] = 1
            pointwise_instances[gt_instances == -1] = 0

        # get labels by taking mean 2d feature over segments of instances
        labels, nonzero_instances = mean_instance_feature(
            segment_featues=segment_feats_2d.cpu().numpy(),
            segmentwise_instances=segmentwise_instances,
        )  # np(n_instances, dim_feat_2d)
        labels = labels[nonzero_instances]
        pointwise_instances = pointwise_instances[:, nonzero_instances]

        # save everything as np arrays
        scan_dir = os.path.join(cfg.ncut.save_dir, scene_name)
        if os.path.exists(scan_dir):
            rmtree(scan_dir)
        os.makedirs(scan_dir, exist_ok=True)
        coords_file = os.path.join(scan_dir, cfg.ncut.coords_filename)
        instances_file = os.path.join(scan_dir, cfg.ncut.instances_filename)
        labels_file = os.path.join(scan_dir, cfg.ncut.labels_filename)
        log.info(f"Number of instances: {labels.shape[0]}")
        np.save(coords_file, coords.cpu().numpy())  # (n_points, 3)
        log.info(f"Saved: {coords_file}")
        np.save(instances_file, pointwise_instances)  # (n_points, n_instances) (onehot)
        log.info(f"Saved: {instances_file}")
        np.save(labels_file, labels)  # (n_instances, feature_dim_2d)
        log.info(f"Saved: {labels_file}")

        # import clip

        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # clip_model, _ = clip.load("ViT-B/32", device=device)
        # query = ["a table in a scene"]
        # query = clip.tokenize(query).to(device)
        # with torch.no_grad():
        #     query_feat = clip_model.encode_text(query).float()
        #     query_feat = F.normalize(query_feat, p=2, dim=1)
        # normalized_segment_feats = F.normalize(segment_feats_2d, p=2, dim=-1)
        # normalized_point_feats = F.normalize(feats_2d, p=2, dim=-1)
        # seg_corr_cols = normalized_segment_feats @ query_feat.T
        # point_corr_cols = normalized_point_feats @ query_feat.T
        # seg_corr_cols = (seg_corr_cols - seg_corr_cols.min()) / seg_corr_cols.max()
        # seg_corr_cols = torch.hstack([seg_corr_cols, seg_corr_cols, seg_corr_cols])
        # point_corr_cols = (
        #     point_corr_cols - point_corr_cols.min()
        # ) / point_corr_cols.max()
        # point_corr_cols = torch.hstack(
        #     [point_corr_cols, point_corr_cols, point_corr_cols]
        # )
        # visualize_3d_feats(
        #     coords.cpu().numpy(),
        #     point_corr_cols.cpu().numpy(),
        #     os.path.join(scan_dir, "point_corr.html"),
        # )
        # visualize_segments(
        #     coords.cpu().numpy(),
        #     unique_segments.cpu().numpy(),
        #     segment_connectivity.cpu().numpy(),
        #     segment_ids.cpu().numpy(),
        #     os.path.join(scan_dir, "segment_corr.html"),
        #     custom_cols=seg_corr_cols.cpu().numpy(),
        # )

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
                segment_connectivity.cpu().numpy(),
                segment_ids.cpu().numpy(),
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
