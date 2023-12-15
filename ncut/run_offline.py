import torch
import hydra
import sys

@hydra.main(
    config_path="../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg):
    sys.path.append(hydra.utils.get_original_cwd())

    from ncut.ncut_datasets import NormalizedCutDataset

    device = torch.device("cuda:0")

    loader = NormalizedCutDataset.dataloader_from_hydra(cfg.ncut.data, only_first=True)
    for sample in loader:
        coords = sample["coords"][0].to(device)
        feats_3d = sample["feats_3d"][0].to(device)
        feats_2d = sample["feats_2d"][0].to(device)
        segment_ids = sample["segment_ids"][0].to(device)
        segment_connectivity = sample["segment_connectivity"][0].to(device)
        print(coords.shape)
        print(feats_3d.shape)
        print(feats_2d.shape)
        print(segment_ids.shape)
        print(segment_connectivity.shape)

if __name__ == "__main__":
    main()