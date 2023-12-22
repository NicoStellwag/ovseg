import hydra
from omegaconf import DictConfig
import os
import sys
import numpy as np
from tqdm import tqdm


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())
    from ovseg.feature_dim_reduction.savable_pca import SavablePCA

    ds = hydra.utils.instantiate(
        cfg.data.train_dataset
    )  # fit only to training samples!

    all_feature_vecs = []
    for i in tqdm(range(len(ds))):
        (_, _, _, instance_feature_vecs, _, _, _, _, _) = ds[i]
        all_feature_vecs.append(instance_feature_vecs)

    all_feature_vecs = np.concatenate(all_feature_vecs, axis=0)

    path = os.path.join(cfg.data.train_dataset.ground_truth_dir, "pca.pkl")
    pca = SavablePCA(n_components=cfg.general.num_targets)
    pca.fit(all_feature_vecs)
    pca.save(path)


if __name__ == "__main__":
    main()
