# Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching

This repository contains a tensorflow implementation for the paper "[Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching](http://www.stat.ucla.edu/~ywu/CoopNets/doc/CoopNets_AAAI.pdf)".
(http://www.stat.ucla.edu/~ywu/CoopNets/main.html)

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scipy](https://www.scipy.org/install.html)
- [pillow](https://pillow.readthedocs.io/en/latest/installation.html)
- [progressbar](http://progressbar-2.readthedocs.io/en/latest/index.html)

## Usage

First, download [dataset](http://www.stat.ucla.edu/~ywu/CoopNets/main.html) and save it to `./data` directory.

To train a model with ***rock*** dataset:

    $ python main.py --num_epochs 200 --d_lr 0.01 --g_lr 0.0001 --category rock --data_dir ./data --batch_size 100 --output_dir ./output
synthesized results will be saved in `./output/rock/synthesis`

To test generator by synthesizing interpolation results with trained model:

    $ python main.py --test --category rock --output_dir ./output --ckpt ./output/rock/checkpoints/model.ckpt --sample_size 144
testing results will be saved in `./output/rock/test`

## Results
**Descriptor result**
![descriptor](assets/descriptor.png)

**Generator result**
![generator](assets/generator.png)

**Interpolation result**
![interpolation](assets/interpolation.png)


## Reference
    @inproceedings{coopnets,
        author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Wu, Ying Nian},
        title = {Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching},
        booktitle = {The 32nd AAAI Conference on Artitifical Intelligence},
        year = {2018}
    }
    
For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu)
