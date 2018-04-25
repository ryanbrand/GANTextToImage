# GANTextToImage
Final project for COMS W4995-2 at Columbia University, Spring 2018

### Primary References
- https://github.com/reedscot/cvpr2016
- https://github.com/reedscot/icml2016
- https://github.com/hanzhanggit/StackGAN-v2

### Dependencies
Ubuntu 14.04 with CUDA 8 and cudNN 6 installed for GPU

Follow these instructions to install Torch with LuaJIT: `http://torch.ch/docs/getting-started.html`

Create anaconda3 environment using `conda create -n <name> --file pytorch_requirements.txt`

`pip install` the following packages in your environment:
- `torch`
- `torchvision`
- `scipy`
- `torchfile`
- `tensorboard==1.0.0a6`
- `python-dateutil`
- `easydict`
- `pandas`

### Testing
python2 GANTextToImage/gan/test.py --cfg="cfg_file" embedding_file_path=""
feed embedding in as embedding_file_path=

