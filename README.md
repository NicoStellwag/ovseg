# ovseg

Unsupervised open vocabulary 3D instance segmentation.

This repo is [Mask3D](https://github.com/JonasSchult/Mask3D) plus a diff, follow the original installation instructions.
There might be some problems with setting up the conda env,
replacing the line with the following worked for me:
```sh
conda create --name mask3d_cuda113 python=3.10.9
conda activate mask3d_cuda113
pip install "cython<3.0.0"
pip install numpy==1.24.2
pip install --no-build-isolation pycocotools==2.0.4
pip install --no-build-isolation "pyyaml<6.0"
# comment pyyaml and pycocotools in the environment yaml!
conda env update -f environment.yml
```

## Converting Mask3D labels to CLIP vectors

Todos:
- Modify semseg dataset to load clip vecs as labels
- Modify models.criterion.SetCriterion.loss_labels to do mean cos dist or something
- We probably want parametric instance queries with dim(query) > dim(clip)
- Think about query initialization
- Modify mask module to return clip vecs instead of class probabilities (if it returs logits we might get away with modifying the # of classes accordingly (probably dim(clip) + 1 for the unassigned class))