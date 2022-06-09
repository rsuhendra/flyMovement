import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
from io import StringIO
import os
from trackImproved_1 import *
import pickle
import sys
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

groupName = str(sys.argv[1])
# 'JennaKir+']#,'JennaWT','61933+']#['22C06Kir','22C06FL50','61933+','61933Kir']#['PFNdKir','PFNd+']#,'Kir2021_+']#,'R4+']#,'SS00096_Kir']#['DNa2_+','A1Kir','P9Kir']#['SS00096_+','Kir2021_+','SS00096_Kir']#['KirFL50jenna','61933+','22C06FL50','22C06Kir']#['61933Kir']
temps = ['30', '35', '40']

# for inputDir in inputDirs:
# 	os.system("python basic_stats_track.py " +inputDir)
# 	os.system("python basic_boxplots.py " +inputDir)

print('Processing '+groupName+' ...')
for j in range(len(temps)):
    inputDir0 = 'outputs/outputs_'+groupName+'/output_'+temps[j]+'/'
    #inputDir0 = 'outputs_'+str(temps[j])+'/output_'+temps[j]+inputDir+'/'
    for fname in os.listdir(inputDir0):
        # skip if trajectory file already exists
        if os.path.exists('traj_plots/traj_'+groupName+'/'+fname.split(".")[0]+'_trajectory_plot.pdf'):
            continue
        if '.output' in fname:
            # print(fname)
            os.system("python plot_traj.py " + inputDir0 +
                        fname + ' traj_' + groupName + ' 0 ' + str(j))
            # os.system("python plot_traj.py " +inputDir0+fname+ ' ' + inputDir + ' 1 ' +str(j))
            # os.system("python plot_traj.py " +inputDir0+fname+ ' ' + inputDir + ' 2 ' +str(j))


# # # # temps = ['30','35','40']
# for inputDir in inputDirs:
# 	for j in range(0,len(temps)):
# 			inputDir0 = 'outputs_'+str(temps[j])+'/output_'+temps[j]+inputDir+'/'
# 			os.system("python boundary_analysis_flex.py " +inputDir0)

# # # # #turn cross
# # # # # inputDirs = ['61933Kir']
# for inputDir in inputDirs:
# 	os.system("python turn_cross_plots.py " +inputDir)
