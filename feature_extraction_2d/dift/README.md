# Diffusion Features (DIFT)
dift feature extactor, works with cuda 11.3 installed and pytorch for cuda 11.7

### [Project Page](https://diffusionfeatures.github.io/) | [Paper](https://arxiv.org/abs/2306.03881) | [Colab Demo](https://colab.research.google.com/drive/1tUTJ3UJxbqnfvUMvYH5lxcqt0UdUjdq6?usp=sharing)


## Prerequisites
```
conda env create -f environment.yml
conda activate dift
```

## Extract DIFT for a given image
You could use the following [command](extract_dift.sh) to extract DIFT from a given image, and save it as a torch tensor. These arguments are set to the same as in the semantic correspondence tasks by default.
```
python extract_dift.py \
    --input_path ./assets/cat.png \
    --output_path dift_cat.pt \
    --img_size 768 768 \
    --t 261 \
    --up_ft_index 1 \
    --prompt 'a photo of a cat' \
    --ensemble_size 8
```
Here're the explanation for each argument:
- `input_path`: path to the input image file.
- `output_path`: path to save the output features as torch tensor.
- `img_size`: the width and height of the resized image before fed into diffusion model. If set to 0, then no resize operation would be performed thus it will stick to the original image size. It is set to [768, 768] by default. You can decrease this if encountering memory issue.
- `t`: time step for diffusion, choose from range [0, 1000], must be an integer. `t=261` by default for semantic correspondence.
- `up_ft_index`: the index of the U-Net upsampling block to extract the feature map, choose from [0, 1, 2, 3]. `up_ft_index=1` by default for semantic correspondence.
- `prompt`: the prompt used in the diffusion model.
- `ensemble_size`: the number of repeated images in each batch used to get features. `ensemble_size=8` by default. You can reduce this value if encountering memory issue.
