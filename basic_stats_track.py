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

# dataFname = "./output/"
# inputDir = str(sys.argv[1])
# arenaData = pickle.load(open(arenaFile,"rb"))
colorOption = 3
behavior = []
behaviorCat = []
color = []
list1 = []
listSF = []
listFramesInTempB = []
listFNames = []
listEventNum = []
posList = []
quadList = []
maxProjList = []
angList = []
ang_val_list = []
af_list = []
ang_sum = []
vel_sum = []
tooShort = []

def calcQuadrant(x_pos, y_pos):
    leftRight = x_pos < y_pos*m1 + b1
    upDown = y_pos < m2*x_pos + b2
    if leftRight and not upDown:
        return 3
    elif leftRight and upDown:
        return 2
    elif not leftRight and upDown:
        return 1
    else:
        return 4

def isQuadHot(quad, state):
    if state == 1:
        return quad == 2.0 or quad == 4.0
    else:
        return quad == 1.0 or quad == 3.0

hotQuads = [1, 3]
coldQuads = [2, 4]
onW = []
onW_inC = []
k = 0
nearThresh_temp = 20
nearThresh = 7
useAll = 0
maxDiff = 300
# read in all behaviors in output directory
fSeqs, allCenters, allMaxes, allStates = [], [], [], []
vList, rvList = [], []
fig, ax = plt.subplots()

count = 0
count2 = 0
countAll = 0
count_move = 0
allDists = []
velList = []
frameRate = 30

groupName = str(sys.argv[1])
temps = ['40', '35', '30', '25']
temps_cold = ['20', '15']
# temps +=temps_cold
dirs=[]
for x in temps:
    dirs.append('outputs/outputs_'+groupName+'/output_'+x+'/')

# run this for all 4 temps.
for inputDir in dirs:
    allHotFracs = []
    allAvSpeeds = []
    all_distTravs = []
    allColdSpeeds = []
    allHotSpeeds = []
    allFnames = []
    for filename in os.listdir(inputDir):
        if filename.split(".")[-1] == 'output':
            ct = 0
            (allEvents, tV, sV, fAngs_deriv, fc_x, fc_y, fAngs, hvCloser, threshVal1,
             scaling, bodyLength, antennaeDist) = pickle.load(open(inputDir+filename, "rb"))
            # get filename, parameters associated
            arenaFile = allEvents[0].arenaFile
            aF = open(arenaFile, "rb")
            data = pickle.load(aF, encoding = 'bytes')
            q4 = list(data[0])
            halfWidth = data[1]/2.
            halfHeight = data[2]/2.
            t1 = np.array(data[3])
            t2 = np.array(data[4])
            aF.close()
            # vid = imageio.get_reader('/home/josh/Desktop/eP/github/flyMovement/WTs/FL50/'+filename.split('.')[0].split('/')[-1]+'.mp4',  'ffmpeg')#for demoing.

            # computations for basic geometry of the arena
            t1 = data[3]
            t2 = data[4]

            m1 = (t1[0][1]-t1[0][0])/(t1[1][1]-t1[1][0])  # 1/slope

            m2 = (t2[1][1]-t2[1][0])/(t2[0][1]-t2[0][0])

            b1 = -m1*t1[1][1] + t1[0][1]
            b2 = -m2*t2[0][1] + t2[1][1]

            intersectCenter = np.array(
                [(m1*b2+b1)/(1.-m1*m2), m2*((m1*b2+b1)/(1.-m1*m2))+b2])

            vertVector = np.array([(t1[0][1]-t1[0][0]), (t1[1][1]-t1[1][0])])
            vertVector = vertVector/np.linalg.norm(vertVector)

            if vertVector[1] < 0:
                vertVector = -vertVector

            horizVector = np.array([(t2[0][1]-t2[0][0]), (t2[1][1]-t2[1][0])])
            horizVector = horizVector/np.linalg.norm(horizVector)
            if horizVector[0] < 0:
                horizVector = -horizVector
            angVert = np.arctan2(vertVector[1], vertVector[0])
            angHoriz = np.arctan2(horizVector[1], horizVector[0])
            angVert = (180*angVert/np.pi) % 360.
            angVert = 180. - angVert  # since image has inverted y axis.
            angHoriz = 180.*angHoriz / \
                np.pi if abs(angHoriz) < (np.pi/2) else 180. * \
                angHoriz/np.pi - 180.
            angHoriz *= -1.

            showQuadrants = int(filename.split(".")[0].split("_")[-1])
            # flipping
            #showQuadrants = 3-showQuadrants

            # calculate basic stats for each video. then add together to get group stats.

            # hot vs cold time spent

            hot_or_not = [isQuadHot(calcQuadrant(fc_y[i1], fc_x[i1]), showQuadrants)
                          for i1 in range(1200, len(fc_x)) if fc_x[i1] > 0]
            hotFrac = np.sum(hot_or_not)/float(len(hot_or_not))
            # print hotFrac
            # print np.mean(hot_or_not)

            # average speed (when moving)
            av_speed = [np.abs(tV[i1])*frameRate/scaling for i1 in range(0, len(tV))
                        if np.abs(tV[i1])*scaling/4.7 > 0.1 and fc_x[i1] > 0]
            av_speed_vid = np.mean(av_speed)

            # average speed in hot/cold
            av_speed_hot = [np.abs(tV[i1])*frameRate/scaling for i1 in range(0, len(tV)) if np.abs(
                tV[i1])*scaling/4.7 > 0.1 and isQuadHot(calcQuadrant(fc_y[i1], fc_x[i1]), showQuadrants) and fc_x[i1] > 0]
            av_speed_vidH = np.mean(av_speed_hot)

            av_speed_cold = [np.abs(tV[i1])*frameRate/scaling for i1 in range(0, len(tV)) if np.abs(
                tV[i1])*scaling/4.7 > 0.1 and not isQuadHot(calcQuadrant(fc_y[i1], fc_x[i1]), showQuadrants) and fc_x[i1] > 0]
            av_speed_vidC = np.mean(av_speed_cold)

            # total distance covered.
            distTraveled = 0
            last_x = fc_x[0]
            last_y = fc_y[0]
            for i1 in range(1, len(fc_x)):
                # as long as no big jump, just sum up the track distance.
                # if (fc_x[i1] - last_x)<5*scaling/4.7 and (fc_y[i1] - last_y)<5*scaling/4.7 and np.abs(tV[i1])*scaling/4.7>0.1:
                # 	distTraveled+= np.sqrt((fc_x[i1] - last_x)**2 +  (fc_y[i1] - last_y)**2)/scaling
                distTraveled += np.abs(tV[i1])/scaling
                # last_x = fc_x[i1]
                # last_y = fc_y[i1]
                # else:
                # 	last_x = fc_x[i1]
                # 	last_y = fc_y[i1]

            allHotFracs.append(hotFrac)
            allAvSpeeds.append(av_speed_vid)
            all_distTravs.append(distTraveled)
            allColdSpeeds.append(av_speed_vidC)
            allHotSpeeds.append(av_speed_vidH)
            allFnames.append(filename)
            # if hotFrac< 0.1:
            # 	print 'check',filename

    if not os.path.exists('./basic_stats/'):
        os.makedirs('./basic_stats/')
    if not os.path.exists('./basic_stats/stats_'+groupName+'/'):
        os.makedirs('./basic_stats/stats_'+groupName+'/')


    f1 = open('basic_stats/stats_'+groupName+'/'+inputDir.split('/')[-2]+'_'+groupName+'_basic_stats.pkl', 'wb')
    pickle.dump((allHotFracs, allAvSpeeds, all_distTravs,
                allColdSpeeds, allHotSpeeds, allFnames), f1)
    f1.close()