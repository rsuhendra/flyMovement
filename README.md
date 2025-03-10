# flyMovement (THIS README IS NOT UPDATED)

This repository is forked from a private repository from user joshuailevy for the publication:

Simões, J.M., Levy, J.I., Zaharieva, E.E. et al. Robustness and plasticity in Drosophila heat avoidance. Nat Commun 12, 2044 (2021). https://doi.org/10.1038/s41467-021-22322-w

I've modified it for my own purposes but most of the code was written originally by (and all credit goes to) https://github.com/joshuailevy. 

## Instructions

YML FILE IS CONTAINED. INSTALL LIKE SO:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

This repository contains functions necessary for high throughput processing of video for the study of the behaving fly in a temperature arena. The tracking code determines the location and orientation of the fly in a robust fashion, which allows for detailed study of behavioral dynamics. Some of the important functions in this repository include:

**trackImproved_1.py**: Tracking script. To run the program on a test video, you just have to run the command:
```
python trackImproved_1.py [location of video] [display mode-default 0] [location of arena file]
```
To plot the turns as they happen in the video, simply change the display mode to 1. If you want to plot the velocities (both translational and angular), just change display mode to 2. 

Returns .output file with name output_[video name].output. 

**run_group.py**: Script for running a bunch of videos with the same conditions (i.e. same lighting, arena, camera) once you've check to make sure things are working well. NOTE: This is currently hard-coded, and needs to be modified for conditions used in the video.

```
python run_group.py [output directory] [directory with video files]
```

**drawArena.py**: Script for determining the dimension of the behavioral arena. Requires a calibration file for the two-choice arena. 

```
python drawArena.py [calibration file] [arena file name]
```

Returns a .arena file, to be used for tracking.I usually store those in a folder called arena. 

**run_all_plots.py**: handy script that runs a whole bunch of different types of analyses in a single go. Provided a folder of tracking output files, it will generate boxplots for AI/walking speed, trajectories for each trial, sideways histogram plots, turn/cross plots and more. Many of these analysis scripts are detailed below.   

```
python run_all_plots.py
```

**boundary_analysis_flex.py**: Script for performing basic analysis at the boundary region, including turns/cross quantification, turn depth over time, and sideways histograms.`

```
python boundary_analysis_flex.py [name of folder containing output files]
```

**staff_plot_flex.py**: Script for plotting boundary events (same as those analyzed in boundary_analysis_flex.py) on the staff plots, where isothermal lines are drawn every 0.5C. Note, there is also **lines_plot_flex.py**, which plots the body axis of the fly at every time point. 

```
python staff_plot_flex.py [name of folder containing output files]
```

**basic_stats_track.py**: Analysis of simple locomotor information. This includes avoidance, average speed, hot speed, cold speed, and distance traveled. NOTE: input should be the name of the given fly as given in your output directory (e.g. FL50 or HCKir). 
 

```
python basic_stats_track.py [name of genotype]
```
**basic_boxplots.py**: Makes boxplots of the information calculated in basic_stats_track.py. 

```
python basic_boxplots.py [name of genotype]
```

**basic_boxplots_stats.py**: Generates statistics for the data analyzed by the basic_boxplots.py script. This requires changing some stuff in the script itself. If you're not sure how to use this, let me know. 
```
python basic_boxplots_stats
```

**findResponse.py**: Analysis for finding first response behaviors in the boundary region. This method will bring up a series of behaviors occuring in the boundary region, colored by their order. To select a behavior, you choose the number of the behavior 1,2,3,4 after closing the plot showing the behavior. If the event is of bad quality (e.g. fly is on the wall or some tracking issue), choose 0. Note: please be careful when using... can be a bit tricky at first. This generates a .response file, which is used as the input to the next function, 

**plot_special_goof**.  

```
python find_response.py [name of folder containing output files]
```

**plot_special_goof.py**: Script to calculate polar plots, turn prediction given temperature difference between antennae plots.

```
python plot_special_goof.py [name of response file generated by find_response.py]
```

**plot_traj.py**: Script to plot full trajectories. 

```
python plot_traj.py [name of output file]   [folder to write to]   [color option (0 =trans vel., 1 = rot vel., 2 = time)]
```

To run it, you'll need python 3.8, as well as the following packages:
## Required packages
* numpy
* opencv
* imageio
* matplotlib
* os
* sys
