import matplotlib
matplotlib.use('Agg')
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

filename = str(sys.argv[1])
# n1 = int(sys.argv[2])
# n2 = int(sys.argv[3])
outFolder = str(sys.argv[2])
colorType = int(sys.argv[3])
quadrant_color = int(sys.argv[4])
showQuadrants = int(filename.split("/")[-1].split(".")[0].split("_")[-1])
# flipping
showQuadrants = 3-showQuadrants
# print filename
ct=0
(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,scaling,bodyLength,antennaeDist) = pickle.load(open(filename,"rb"))
#get filename, parameters associated
arenaFile = allEvents[0].arenaFile
aF = open(arenaFile,"rb")
data = pickle.load(aF, encoding='bytes')
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

m1 = (t1[0][1]-t1[0][0])/(t1[1][1]-t1[1][0]) # 1/slope

m2 = (t2[1][1]-t2[1][0])/(t2[0][1]-t2[0][0])


b1 = -m1*t1[1][1] +t1[0][1]
b2 = -m2*t2[0][1] + t2[1][1]

intersectCenter = np.array([(m1*b2+b1)/(1.-m1*m2),m2*((m1*b2+b1)/(1.-m1*m2))+b2])
q4 = intersectCenter
# q4 = [q4[1],q4[0]]
vertVector = np.array([(t1[0][1]-t1[0][0]),(t1[1][1]-t1[1][0])])
vertVector = vertVector/np.linalg.norm(vertVector)

if vertVector[1]<0:
	vertVector = -vertVector

horizVector = np.array([(t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0])])
horizVector = horizVector/np.linalg.norm(horizVector)
if horizVector[0]<0:
	horizVector = -horizVector

# angVert =np.arctan2(vertVector[1],vertVector[0])
# angHoriz = np.arctan2(horizVector[1],horizVector[0])
# angVert = (180*angVert/np.pi) % 360.
# angVert = 180. - angVert # since image has inverted y axis. 
# angHoriz = 180.*angHoriz/np.pi if abs(angHoriz) < (np.pi/2) else 180.*angHoriz/np.pi -180.
# angHoriz*=-1.

tH1 = 180./np.pi*np.arctan2(horizVector[1],horizVector[0])
tV1 =180./np.pi*np.arctan2(vertVector[1],vertVector[0])
tV1 = tV1 % 360.
tH1 = tH1 if abs(tH1) < (90.) else tH1 -180.
#find sequences of behaviors in which the fly is in the desired band
# for checking arena parameters. 
# ax11.plot(t1[0],t1[1],color='blue')
# ax11.plot(t2[0],t2[1],color='green')
# ax11.scatter(intersectCenter[0],intersectCenter[1],color='red')
# showQuadrants = int(filename.split(".")[0].split("_")[-1])
last = False
cSeq = []
currentVs = []

from matplotlib.patches import Wedge
l = 0
# cT =len(nums)*[(0,0,0,0)]
inTurn = 0
img = None
figScat,ax = plt.subplots()
n1 = 0
n2 = len(fc_y)
nums = range(n1,n2)
# nums = xrange(0,len(fc_y))

for num in nums:
	if num>0:
		if fc_y[num]==0 and fc_y[num-1]!=0:
			fc_y[num] = fc_y[num-1]
			fc_x[num] = fc_x[num-1]
			fAngs[num] = fAngs[num-1]
		elif fc_y[num]==0:
			fc_y[num] = np.nan
			fc_x[num] = np.nan
			fAngs[num] = np.nan

if quadrant_color ==0:
	alpha = 0.3
else:
	alpha = 0.6
# ax.plot(t1[0],t1[1],color='blue')
# ax.plot(t2[0],t2[1],color='green')
# ax.scatter(intersectCenter[0],intersectCenter[1],color='red')
if showQuadrants ==1:
	topRightQuad = Wedge(q4,(halfWidth+halfHeight)/2+5,tH1,tV1,alpha =alpha,color = 'red')
	bottomLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2+5,180+tH1,180+tV1,alpha =alpha,color = 'red')
	ax.add_patch(topRightQuad)
	ax.add_patch(bottomLeftQuad)
elif showQuadrants ==2:
	topLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2+5,tV1,180+tH1,alpha =alpha,color = 'red')
	bottomRightQuad = Wedge(q4,(halfWidth+halfHeight)/2+5,180+tV1,360+tH1,alpha =alpha,color = 'red')
	ax.add_patch(topLeftQuad)
	ax.add_patch(bottomRightQuad)
# scat1 = ax.scatter([],[],s=8)
if colorType == 0:#trans vel
	scat1 =ax.scatter(fc_y[n1:n2]-7*np.sin(np.pi/180.*fAngs[n1:n2]),fc_x[n1:n2]+7*np.cos(np.pi/180*fAngs[n1:n2]),s=10,c=np.abs(tV/scaling*30),vmin=0,vmax=10,zorder=1000,cmap=plt.cm.jet)
elif colorType==1:#rot vel
	import matplotlib.colors as colors
	def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
		return new_cmap
	cmap = plt.get_cmap('PRGn_r')
	new_cmap = truncate_colormap(cmap, 0.15, 0.85)
	scat1 =ax.scatter(fc_y[n1:n2]-7*np.sin(np.pi/180.*fAngs[n1:n2]),fc_x[n1:n2]+7*np.cos(np.pi/180*fAngs[n1:n2]),s=10,c=fAngs_deriv*30,vmin=-150,vmax=150,zorder=1000,cmap=new_cmap)
if colorType ==2:#sidestep vel
	scat1 =ax.scatter(fc_y[n1:n2]-7*np.sin(np.pi/180.*fAngs[n1:n2]),fc_x[n1:n2]+7*np.cos(np.pi/180*fAngs[n1:n2]),s=10,c=sV/scaling*30,vmin=-5,vmax=5,zorder=1000,cmap=plt.cm.BrBG)
elif colorType==3:#time
	scat1 =ax.scatter(fc_y[n1:n2]-7*np.sin(np.pi/180.*fAngs[n1:n2]),fc_x[n1:n2]+7*np.cos(np.pi/180*fAngs[n1:n2]),s=10,c=np.arange(len(fc_y[n1:n2])),zorder=1000,cmap=plt.cm.Spectral_r)
plt.gca().invert_yaxis()
ax.set_aspect('equal')
figScat.colorbar(scat1,ax=ax)
# plt.savefig('tester.svg')

if not os.path.exists('./traj_plots/'):
    os.makedirs('./traj_plots/')
if not os.path.exists('./traj_plots/'+outFolder+'/'):
    os.makedirs('./traj_plots/'+outFolder+'/')
	
plt.savefig('traj_plots/' + outFolder +'/'+filename.split('/')[-1].split('.')[0]+'_'+str(colorType)+'_trajectory_plot.pdf')
plt.close()

