# flyMovement (THIS README IS NOT UPDATED)

YML FILE IS CONTAINED. INSTALL LIKE SO:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

This repository contains functions necessary for high throughput processing of video for the study of the behaving fly in a temperature arena. The tracking code determines the location and orientation of the fly in a robust fashion, which allows for detailed study of behavioral dynamics. Some of the important functions in this repository include:

**trackImproved_1.py**: Tracking script. To run the program on a test video, you just have to run the command:
```
python trackImproved_1.py [location of video] [display mode-default 0] [location of arena file]
```
To plot the turns as they happen in the video, simply change the display mode to 1. If you want to plot the velocities (both translational and angular), just change display mode to 2. 

Returns .output file with name output_[video name].output. 

**tsne_param_rank.py**: Parametric t-SNE embedding of tracking data.
```
python tsne_param_rank.py [folder with tracking output files] [edge parameter]
```
The edge parameter is chosen to be 1 for studying behavior away from the arena edges, 0 only near the edges, and anything else for all behavior. This returns a .h5 and .params file, which will be used in the run_tsne_model.py script. 

**run_tsne_model.py**: Script to map points through parametric mapping. 
``` 
python run_tsne_model.py [folder with tracking output files] [model file (.h5)]
```
This will return a .json file that contains the 2D positions of all behaviors in the embedded space,  as well plots a kernel density estimate of the space, and a segmentation similar to that shown in Berman et al. (2014). 

**check_groups.py**: Script to find symmetric segmentation of behavioral space using Gaussian Mixture Modelling. 
```
python check_groups.py [folder with tracking output files] [model file (.h5)] 
```
This returns a .gmm file, which can be used in the gmm_stats.py, internal_symmetry_comp_gmm.py, and double_comp_gmm_stats.py scripts. 

**gmm_stats.py**: Calculation of significant change in behavioral expression relative to training set via GMM segmentation. 

```
python gmm_stats.py [folder with tracking output files] [model file (.h5)] [master .gmm file]
```
Returns an image describing the regions of increased and decreased behavioral expression. Statistical significance is determined at a multiple hypothesis corrected p-value of 0.05 using Fisher's exact test and Benjamini-Hochberg correction. 

**internal_symmetry_comp_gmm.py**: Comparison of behavior to  symmetric partner (by mirroring through the parametric map) and using segmentation from GMM. 

```
python internal_symmetry_comp_gmm.py [folder with tracking output files] [model file (.h5)] [master .gmm file]
```
Returns an image much like gmm_stats.py. 

**double_comp_gmm_stats.py**: Comparison of two behavioral conditions using GMM segmentation. 

```
python double_comp_gmm_stats.py [folder of group 1 files] [folder of group 2 files] [model file (.h5)] [master .gmm file]
```
Returns image like gmm_stats.py. 

## Required packages
* numpy
* opencv
* imageio
* matplotlib
* os
* sys
