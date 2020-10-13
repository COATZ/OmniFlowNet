# OmniFlowNet
OmniFlowNet

This repository (https://github.com/COATZ/OmniFlowNet) is the offical release of OmniLiteFlowNet for my paper OmniFlowNet: a Perspective Neural Network Adaptation for OpticalFlow Estimation in Omnidirectional Images in ICPR 2020.

OmniFlowNet is an adaptation of CAFFE based CNNs for optical flow estimation in omnidirectional images. This implentation was tested on FlowNet2 (https://github.com/lmb-freiburg/flownet2) and LiteFlowNet2 (https://github.com/twhui/LiteFlowNet2).

# Installation and Run

1. Change the file /src/caffe/util/im2col.cu in the Caffe sources folder.
2. Rebuild the caffe bin and libs.
3. Load the pre-trained weights and deploy file.
4. Select the CAFFE engine for the convolution layers in the deploy file.
5. Run the CNN.

# Performances

A video shows differences between LiteFlowNet2 and OmniFlowNet on several equirectangular scenes: http://www.i3s.unice.fr/~allibert/Videos/icpr20_video.mp4.

# References
