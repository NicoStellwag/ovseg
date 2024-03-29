import MinkowskiEngine as ME
import numpy as np
import torch
from random import random


class OpenVocabVoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
        export=False,
        iou_threshold=0.5
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold
        self.export = export
        self.iou_threshold = iou_threshold

        self.num_queries = num_queries

    def __call__(self, batch):
        if ("train" in self.mode) and (self.small_crops or self.very_small_crops):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            num_queries=self.num_queries,
            export=self.export,
            iou_threshold=self.iou_threshold
        )


class VoxelizeCollateMerge:
    """
    NOT ADPATED!
    """

    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        scenes=2,
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        make_one_pc_noise=False,
        place_nearby=False,
        place_far=False,
        proba=1,
        probing=False,
        task="instance_segmentation",
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba
        self.probing = probing

    def __call__(self, batch):
        if (
            ("train" in self.mode)
            and (not self.make_one_pc_noise)
            and (self.proba > random())
        ):
            if self.small_crops or self.very_small_crops:
                batch = make_crops(batch)
            if self.very_small_crops:
                batch = make_crops(batch)
            if self.batch_instance:
                batch = batch_instances(batch)
            new_batch = []
            for i in range(0, len(batch), self.scenes):
                batch_coordinates = []
                batch_features = []
                batch_labels = []

                batch_filenames = ""
                batch_raw_color = []
                batch_raw_normals = []

                offset_instance_id = 0
                offset_segment_id = 0

                for j in range(min(len(batch[i:]), self.scenes)):
                    batch_coordinates.append(batch[i + j][0])
                    batch_features.append(batch[i + j][1])

                    if j == 0:
                        batch_filenames = batch[i + j][3]
                    else:
                        batch_filenames = batch_filenames + f"+{batch[i + j][3]}"

                    batch_raw_color.append(batch[i + j][4])
                    batch_raw_normals.append(batch[i + j][5])

                    # make instance ids and segment ids unique
                    # take care that -1 instances stay at -1
                    batch_labels.append(
                        batch[i + j][2] + [0, offset_instance_id, offset_segment_id]
                    )
                    batch_labels[-1][batch[i + j][2][:, 1] == -1, 1] = -1

                    max_instance_id, max_segment_id = batch[i + j][2].max(axis=0)[1:]
                    offset_segment_id = offset_segment_id + max_segment_id + 1
                    offset_instance_id = offset_instance_id + max_instance_id + 1

                if (len(batch_coordinates) == 2) and self.place_nearby:
                    border = batch_coordinates[0][:, 0].max()
                    border -= batch_coordinates[1][:, 0].min()
                    batch_coordinates[1][:, 0] += border
                elif (len(batch_coordinates) == 2) and self.place_far:
                    batch_coordinates[1] += (
                        np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                    )
                new_batch.append(
                    (
                        np.vstack(batch_coordinates),
                        np.vstack(batch_features),
                        np.concatenate(batch_labels),
                        batch_filenames,
                        np.vstack(batch_raw_color),
                        np.vstack(batch_raw_normals),
                    )
                )
            # TODO WHAT ABOUT POINT2SEGMENT AND SO ON ...
            batch = new_batch
        elif ("train" in self.mode) and self.make_one_pc_noise:
            new_batch = []
            for i in range(0, len(batch), 2):
                if (i + 1) < len(batch):
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    batch[i][2],
                                    np.full_like(batch[i + 1][2], self.ignore_label),
                                )
                            ),
                        ]
                    )
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    np.full_like(batch[i][2], self.ignore_label),
                                    batch[i + 1][2],
                                )
                            ),
                        ]
                    )
                else:
                    new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
            batch = new_batch
        # return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode)
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
        )


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch


def voxelize(
    batch,
    ignore_label,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
    export,
    iou_threshold
):
    (
        coordinates,
        features,
        labels,
        instance_feature_vecs,
        original_labels,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
        new_instances,
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []

    # calculate sparse coordinates for each sample
    for sample in batch:
        idx.append(sample[8])
        original_coordinates.append(sample[7])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[5])
        original_normals.append(sample[6])
        new_instances.append(sample[9])

        # instance ids are left unchanged throughout voxelize collate,
        # so we can just pass the instance's clip vecs through the target
        # without any modification
        instance_feature_vecs.append(torch.from_numpy(sample[3]).type(torch.float))

        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())

    # Concatenate sample-wise sparse coords and feats to single input for batch sparse tens
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

    # return without further processing for probing
    if probing:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
            ),
            labels,
        )

    # turn labels (semantic class ids) into consecutive sequence starting at 0
    # if train mode do the same for segment ids
    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(
                input_dict["labels"][i][:, 0],
                return_index=True,
                return_inverse=True,
            )
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, -1],
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                input_dict["segment2label"].append(
                    input_dict["labels"][i][ret_index][:, :-1]
                )

    if "labels" in input_dict:
        list_labels = input_dict["labels"]

        target = []
        target_full = []

        # for 1D labels (semantic segmentation) target is just
        # {
        #    "labels": unique labels ids,
        #    "masks": mask for each label id
        # }
        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id] == label_ids.unsqueeze(1),
                        "instance_feats": instance_feature_vecs[batch_id],
                    }
                )
        else:
            # for 3D labels in test mode point2segment at full and sparse
            # resolution is just semantic class of points
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append({"point2segment": input_dict["labels"][i][:, 0]})
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long()
                        }
                    )
            # for 3D labels in train mode target also contains classwise
            # or instance wise masks at segment and point resolution
            else:
                target = get_instance_masks(
                    list_labels,
                    instance_feature_vecs,
                    list_segments=input_dict["segment2label"],
                    task=task,
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                    new_instances=new_instances,
                    iou_threshold=iou_threshold
                )
                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]
                if "train" not in mode or export:
                    target_full = get_instance_masks(
                        [torch.from_numpy(l) for l in original_labels],
                        instance_feature_vecs,
                        task=task,
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                        new_instances=new_instances,
                        iou_threshold=iou_threshold
                    )
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(
                            original_labels[i][:, 2]
                        ).long()
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    if "train" not in mode or export:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                target_full,
                original_colors,
                original_normals,
                original_coordinates,
                idx,
                #new_instances
            ),
            target,
            [sample[4] for sample in batch],
        )
    else:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
                full_res_coords,
                #new_instances
            ),
            target,
            [sample[4] for sample in batch],
        )

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou


def get_instance_masks(
    list_labels,
    instance_feature_vecs,
    task,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
    new_instances=[],
    iou_threshold=0.1
):
    """
    split a list of labels [(n_points, 3), ...]
    to a list of dicts (one for each sample)
    containg the following entries:
    {
        "labels": tens(num_unique_labels,),
        "masks": tens(num_unique_labels, n_points),
        "segment_mask": tens(num_unique_labels, n_segments)
    }
    """
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        filtered_instance_feature_vecs = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1]#.unique()
        instance_ids = instance_ids.type(torch.torch.IntTensor).unique()

        for instance_id in instance_ids:
            # -1 is non-instance id
            if instance_id == -1:
                continue

            # filter out some semantic classes that can be specified
            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            tmp = list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id]
            label_id = tmp[0, 0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            filtered_instance_feature_vecs.append(
                instance_feature_vecs[batch_id][instance_id]
            )
            label_ids.append(label_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            # also create segment level masks if necessary
            if list_segments:
                segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                segment_mask[
                    list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id][
                        :, 2
                    ].unique()
                ] = True
                segment_masks.append(segment_mask)                 



        if len(label_ids) == 0:
            return list()
            

        if list_segments:
            if(len(new_instances[batch_id]) != 0):
                #dirty but does the job
                iou = -1
                mismatched = 0
                new_masks, new_features = new_instances[batch_id]
                for i, mask in enumerate(new_masks):
                    for segment_mask in segment_masks:
                        if(mask.shape != segment_mask.shape):
                            mismatched+=1
                            continue
                        iou = calculate_iou(mask, segment_mask)
                        if(iou >= iou_threshold):
                            segment_masks.append(segment_mask)
                            filtered_instance_feature_vecs.append(new_features[i])
                            break
                    if(iou >= iou_threshold):
                        break
                # if(mismatched != 0):
                #     print(mismatched)

                #better
                # pairwise_overlap = segment_masks @ new_instances[batch_id].T
                # normalization = pairwise_overlap.max(axis=0) + 1e-6
                # norm_overlaps = pairwise_overlap / normalization
                # thresholded = norm_overlaps > iou_threshold


            segment_masks = torch.stack(segment_masks)

        # stack
        filtered_instance_feature_vecs = torch.stack(filtered_instance_feature_vecs)
        label_ids = torch.stack(label_ids)
        masks = torch.stack(masks)


        # if mode is semantic segmentation aggregate all masks of the same semantic class
        # if mode is instance segmentation keep them separate
        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = label_ids == label_id

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(
                        segment_masks[masking, :].sum(dim=0).bool()
                    )

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)

                target.append(
                    {
                        "labels": label_ids,
                        "masks": masks,
                        "segment_mask": segment_masks,
                        "instance_feats": filtered_instance_feature_vecs,
                    }
                )
            else:
                target.append(
                    {
                        "labels": label_ids,
                        "masks": masks,
                        "instance_feats": filtered_instance_feature_vecs,
                    }
                )
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append(
                    {
                        "labels": l,
                        "masks": masks,
                        "segment_mask": segment_masks,
                        "instance_feats": filtered_instance_feature_vecs,
                    }
                )
            else:
                target.append(
                    {
                        "labels": l,
                        "masks": masks,
                        "instance_feats": filtered_instance_feature_vecs,
                    }
                )
    return target


def make_crops(batch):
    """
    split a batch of full scenes into a new batch
    containing quadrants of scnenes as new samples.
    new batch size is at most 4 * old batch size
    """
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        full_res_coords=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
        #new_instances=None
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
        #self.new_instances=new_instances


class NoGpuMask:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        masks=None,
        labels=None,
        #new_instances=None
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels
        #self.new_instances=new_instances
