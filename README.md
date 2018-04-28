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

### Training (Embedding Network):

1. Download the [birds](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE)
 and [flowers](https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM) data.
2. Modify the training script (e.g. `embedding/scripts/train_cub_hybrid.sh` for birds) to point to your data directory.
3. Navigate to `embedding/` and run the training script: `./scripts/train_cub_hybrid.sh`

### Evaluation (Embedding Network):

1. Train a model (see above).
2. Modify the eval bash script (e.g. `eval_cub_cls.sh` for birds) to point to your saved checkpoint.
3. Navigate to `embedding/` and run the eval script: `./scripts/eval_cub_cls.sh`

### Training (Conditional GAN):

1. Download the [birds](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing) and [flowers](https://drive.google.com/file/d/0B0ywwgffWnLLMl9uOU91MV80cVU/view?usp=sharing) caption data in Torch format and save to `data/`.  You may replace `char-CNN-RNN.pkl` with your own trained embeddings if saved to a pickle file of the same format (Note that if the file does not have the same name, other scripts will need to be modified as well). 
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102) image data and save to `data/birds/` and `data/flowers/`, repectively.
3. Navigate to `gan/` and train a hierarchical model on the bird (CUB) or flower (Oxford-102) datasets using preprocessed embeddings by running the following:
  -  `python main.py --cfg cfg/<bird_config.yml> --gpu 0`
  -  `python main.py --cfg cfg/<flower_config.yml> --gpu 0`
  
### Evaluation (Conditional GAN):

1. Train a model (see above)
2. Modify `gan/cfg/eval_<dataset>.yml` file to point to the saved checkpoint for your model.
3. Navigate to `gan/` and evaluate a hierarchical model on the bird (CUB) or flower (Oxford-102) datasets using pretrained network by running the following:
  - `python main.py --cfg cfg/eval_birds.yml --gpu 0` to generate samples from captions in birds validation set.
  - `python main.py --cfg cfg/eval_flowers.yml --gpu 0` to generate samples from captions in flowers validation set.
  
### Running Bird Text-To-Image Synthesis Server

1. Train or download both an embedding model and a conditional GAN model for the CUB birds dataset (see above)
2. Modify `test_scripts/test_birds_server.sh` to point to desired embedding network and `gan/cfg/eval_birds.yml` to point to the saved checkpoint you wish to use for the GAN.
3. Run `python server.py` and access server at `<Your IP Address>:6007`.

