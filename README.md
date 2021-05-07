Curious Representation Learning for Embodied Intelligence
==============================

This is the pytorch code for the paper [Curious Representation Learning for Embodied Intelligence](https://arxiv.org/pdf/2105.01060.pdf). This codebase is based on the codebase from Habitat-lab, please see [HABITAT\_README.md](https://github.com/yilundu/crl/blob/master/HABITAT_README.md) for installation instructions for the repository.

## Interactive Pretraining of Embodied Agents

To pretrain agent weights on Matterport3D, please use the following command:

```
python habitat_baselines/run.py --run-type=train --exp-config habitat_baselines/cvpr_config/pretrain/curiosity_pointnav_pretrain.yaml
```

The other configs used in the paper may also be found in habitat\_baselines/cvpr\_config/pretrain.


## Downstream ImageNav Pretraining

To finetune weights on ImageNav, please use the following command: 

```
python habitat_baselines/run.py --run-type=train --exp-config habitat_baselines/cvpr_config/imagenav/curiosity_pointnav_gibson_imagenav.yaml
```

## Downstream ObjectNav Pretraining

To finetune weights on ObjectNav, please use the following command: 

```
python habitat_baselines/run.py --run-type=train --exp-config habitat_baselines/cvpr_config/objectnav/curiosity_pointnav_mp3d_objectnav.yaml
```

## Pretrained Weights

The pretrained CRL model from the Matterport3D environment can be downloaded from [here](https://www.dropbox.com/s/gwxm4x4a1fnloz2/curiosity_pointnav_pretrain.16.pth?dl=0)

## Citing Our Paper

If you find our code useful for your research, please consider citing the following [paper](https://arxiv.org/pdf/2105.01060.pdf), as well as papers included in [HABITAT\_README.md](https://github.com/yilundu/crl/blob/master/HABITAT_README.md).

```	
@article{du2021curious,
    author = {Du, Yilun and Gan, Chuang and
    Isola, Phillip},
    title = {Curious Representation Learning
    for Embodied Intelligence},
    journal = {arXiv preprint arXiv:2105.01060},
    year = {2021}
}
```
