Individial Project

The objective of this project is to put oriented bounding boxes on vehicles in a real street LiDAR scene in Hong Kong. The detector should be invariant to partial occlusion, deformation, noises, and varying degrees of point density - all of which are ubiquitous caveats in LiDAR clouds.

The dataset can be browsed **[here](https://dord.mynetgear.com:5001/static/potree/index.html)**. Click the **'Colour by classification'** button on the left to see the ground truth segmentation.

In case you are wondering, I made the dataset myself. The points have only INTENSITY values and no RGB values.
# Auxiliary software
* MeshLab
* CloudCompare
* lastile
* [labelCloud](https://github.com/ch-sa/labelCloud) *for labelling vehicles*
* [PyTorch-Geometric](https://github.com/pyg-team/pytorch_geometric) **needed by the custom algorithm**

# Auxiliary hardware
* RIEGL

# References
1. Yan, Y., Mao, Y., & Li, B. (2018, October 6). *SECOND: Sparsely embedded convolutional detection*. MDPI. Retrieved February 17, 2022, from https://www.mdpi.com/1424-8220/18/10/3337
2. Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019, May 7). *PointPillars: Fast encoders for object detection from point clouds*. arXiv.org. Retrieved February 17, 2022, from https://arxiv.org/abs/1812.05784
3. Qi, C. R., Litany, O., He, K., & Guibas, L. J. (2019, August 22). *VoteNet: Deep Hough voting for 3D object detection in point clouds*. arXiv.org. Retrieved February 17, 2022, from https://arxiv.org/abs/1904.09664
4. Pyg-Team. (n.d.). PyG-team/pytorch_geometric: *Graph Neural Network Library for PyTorch*. GitHub. Retrieved February 17, 2022, from https://github.com/pyg-team/pytorch_geometric