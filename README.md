## Segmentation of Large Public Spaces Point Clouds

(Individial Project)

### Demo video: https://www.youtube.com/watch?v=YKtg5CZMUmM

### Note: Due to the file size limit of GitHub, some of the resources or checkpoints can be found here: https://dord.mynetgear.com/pointcloudproject/

There are two parts (and hence datasets) of the project: outdoors and indoors.

For outdoors, the dataset can be browsed **[here](https://dord.mynetgear.com:5001/static/potree/index.html)**. Click the **'Colour by classification'** button on the left to see the ground truth segmentation. In case you are wondering, I made the dataset myself. The points have only INTENSITY values and no RGB values.

For indoors, we use the S3DIS dataset, which is composed of coloured point clouds of rooms and corridors in different styles. 

In this project, besides using the PAConv [1] model from the MMDetection3D library (the library was contrived by my alma mater CUHK), we propose a custom method by building a graph neural network (GNN) using k-nearest neighbour (KNN) graphs. An edge is present between two points if it is among its k nearest neighbours. It is inspired by PointNet++ [2] but is built from stratch, layer by layer, module by module. For the custom method, we also experimented with transformers. Both models are built with the help of graph-specific operators provided by the PyTorch-geometric extension library. [3]

### Auxiliary software
* MeshLab
* CloudCompare
* lastile (for dataset tiling)
* lasview & laslayers (for labelling point clouds)
* las2las & lasmerge (for dataset augmentation)
<!-- * [labelCloud](https://github.com/ch-sa/labelCloud) *for labelling vehicles* -->
* [PyTorch-Geometric](https://github.com/pyg-team/pytorch_geometric) **needed by the custom algorithm** [4]
* Tensorboard (for logging and monitoring)

### Auxiliary hardware
* RIEGL

### References
<!--
1. Yan, Y., Mao, Y., & Li, B. (2018, October 6). *SECOND: Sparsely embedded convolutional detection*. MDPI. Retrieved February 17, 2022, from https://www.mdpi.com/1424-8220/18/10/3337
2. Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019, May 7). *PointPillars: Fast encoders for object detection from point clouds*. arXiv.org. Retrieved February 17, 2022, from https://arxiv.org/abs/1812.05784
3. Qi, C. R., Litany, O., He, K., & Guibas, L. J. (2019, August 22). *VoteNet: Deep Hough voting for 3D object detection in point cluds*. arXiv.org. Retrieved February 17, 2022, from https://arxiv.org/abs/1904.09664
-->

1. Xu, M., Ding, R., Zhao, H., & Qi, X. (2021, April 26). *PAConv: Position adaptive convolution with dynamic kernel assembling on point clouds*. arXiv.org. Retrieved February 17, from https://arxiv.org/abs/2103.14635 
2. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017, June 7). *PointNet++: Deep hierarchical feature learning on point sets in a metric space*. arXiv.org. Retrieved February 17, from https://arxiv.org/abs/1706.02413 
3. Fey, M., & Lenssen, J. E. (2019, April 25). *Fast graph representation learning with PyTorch Geometric*. arXiv.org. Retrieved February 17, from https://arxiv.org/abs/1903.02428
4. Pyg-Team. (2019). *PyG-team/pytorch_geometric: Graph Neural Network Library for PyTorch*. GitHub. Retrieved February 17, 2022, from https://github.com/pyg-team/pytorch_geometric
