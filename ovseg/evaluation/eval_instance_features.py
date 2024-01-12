import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import os
import numpy as np
import sys
import clip
from tqdm import tqdm


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    c_fn = hydra.utils.instantiate(cfg.data.validation_collation)
    ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    loader = hydra.utils.instantiate(
        cfg.data.validation_dataloader,
        dataset=ds,
        collate_fn=c_fn,
    )

    class_cos_sims = {}
    class_counts = {}

    for batch in tqdm(loader):
        _, target, _ = batch

        instance_features = target[0]["instance_feats"]
        instance_features = F.normalize(instance_features, p=2, dim=-1)

        labels = target[0]["labels"]
        labels[labels == 0] = -1
        class_ids = ds._remap_model_output(labels + ds.label_offset)

        keep = class_ids != 255
        instance_features = instance_features[keep]
        class_ids = class_ids[keep]

        label_features = ds.map2features(
            class_ids, instance_features.device
        )  # already normed

        cos_sims = (instance_features * label_features).sum(-1)
        cos_sims = cos_sims.cpu().numpy()

        for i, c_id in enumerate(class_ids):
            if np.isclose(cos_sims[i], 0):
                print(
                    f"0 for label feat sum {label_features[i].sum()} and instance feat sum {instance_features[i].sum()}"
                )
            if c_id not in class_counts:
                class_counts[c_id] = 1
                class_cos_sims[c_id] = cos_sims[i]
            else:
                class_counts[c_id] += 1
                class_cos_sims[c_id] += cos_sims[i]

    for c_id in class_counts:
        mean_cos_sim = class_cos_sims[c_id] / class_counts[c_id]
        print(f"{c_id: <10}{ds.label_info[c_id]['name']: <30}{mean_cos_sim: <10.4f}")


if __name__ == "__main__":
    main()
