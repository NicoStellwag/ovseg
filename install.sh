#conda create --name mask3d_cuda113 python=3.10.9
#conda activate mask3d_cuda113
#pip install "cython<3.0.0"
#pip install numpy==1.24.2
#pip install --no-build-isolation pycocotools==2.0.4
#pip install --no-build-isolation "pyyaml<6.0"
#conda env update -f environment.yml
#pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
#pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
#
#cd ../detectron2
#python -m pip install -e .
#mkdir third_party
#cd third_party
#
#git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2
pip install torchmetrics==0.11.4
