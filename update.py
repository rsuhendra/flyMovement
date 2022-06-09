import peakutils
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
import os
from io import StringIO
from PIL import Image
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import glob

groups=['DilpLexA_AopKir']

for groupName in groups:
    print('Updating ', groupName)
    currentFiles=[f.split(".")[0] for f in os.listdir("videos/"+groupName)]
    oldFiles = glob.glob("outputs/outputs_"+groupName + '/**/*.output', recursive=True)
    removeCounter=0
    for f in oldFiles:
        if f.split("/")[-1].split(".")[0] not in currentFiles:
            print('Removing ', f.split("/")[-1].split(".")[0])
            removeCounter+=1
            os.remove(f)
            os.remove(f.split('.')[0]+'.stats')
            try:
                os.remove('traj_plots/traj_'+groupName+'/'+f.split("/")[-1].split(".")[0]+'_0_trajectory_plot.pdf')
            except: 
                pass

    print('Removed '+str(removeCounter)+' old output files...')
    oldF = [f.split("/")[-1].split(".")[0] for f in oldFiles]
    addCounter=0
    os.system("python run_group.py videos/"+groupName)
    for f in currentFiles:
        if f not in oldF:
            addCounter+=1
    print('Added in '+str(addCounter)+' new output files...')