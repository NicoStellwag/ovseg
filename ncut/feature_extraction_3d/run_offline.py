import hydra
import torch
import torch.nn as nn
import sys
from omegaconf import DictConfig

import torch


def get_feature_extractor(cfg_fe3d: DictConfig) -> nn.Module:
    model = hydra.utils.instantiate(cfg_fe3d.model)
    model.load_state_dict(torch.load(cfg_fe3d.model.pretrained_weights)["state_dict"])
    return model


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    sys.path.append(hydra.utils.get_original_cwd()) # necessary for hydra object instantiation when script is not in project root
    cfg_fe3d = cfg.ncut.feature_extraction_3d

    model = get_feature_extractor(cfg_fe3d)
    print(len(list(model.parameters())))


if __name__ == "__main__":
    main()
