# OmniFlowNet
OmniFlowNet

This repository (https://github.com/COATZ/OmniFlowNet) is the offical release of OmniFlowNet for my paper "OmniFlowNet: a Perspective Neural Network Adaptation for OpticalFlow Estimation in Omnidirectional Images in ICPR 2020".

OmniFlowNet is an adaptation of CAFFE based CNNs for optical flow estimation in omnidirectional images. This implentation was tested on FlowNet2 (https://github.com/lmb-freiburg/flownet2) and LiteFlowNet2 (https://github.com/twhui/LiteFlowNet2).

# Installation and Run

1. Change the file /src/caffe/util/im2col.cu in the Caffe sources folder.
2. Rebuild the caffe bin and libs.
3. Load the pre-trained weights and deploy file.
4. Select the CAFFE engine for the convolution layers in the deploy file.
5. Run the CNN.

# Performances

A video shows differences between LiteFlowNet2 and OmniFlowNet on several equirectangular scenes: http://www.i3s.unice.fr/~allibert/Videos/icpr20_video.mp4.

# Dataset

The omnidirectional dataset used to test the CNN is available at: https://www.i3s.unice.fr/~allibert/data/OMNIFLOWNET_DATASET.zip.
For the three scenes (Cartoon Tree, Forest and LowPolyModels), the equirectangular rgb images are available as well as the ground truth optical flow.

# References
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:
````
@inproceedings{artizzu:hal-02968191,
  TITLE = {{OmniFlowNet: a Perspective Neural Network Adaptation for Optical Flow Estimation in Omnidirectional Images}},
  AUTHOR = {Artizzu, Charles-Olivier and Zhang, Haozhou and Allibert, Guillaume and Demonceaux, C{\'e}dric},
  URL = {https://hal.archives-ouvertes.fr/hal-02968191},
  BOOKTITLE = {{25th International Conference on Pattern Recognition (ICPR)}},
  ADDRESS = {Milan, Italy},
  YEAR = {2021},
  MONTH = Jan,
  PDF = {https://hal.archives-ouvertes.fr/hal-02968191/file/ICPR_2020_FINAL.pdf},
  HAL_ID = {hal-02968191},
  HAL_VERSION = {v1},
}
````
