import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
import os
import csv
import io
from PIL import Image
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import pickle
import matplotlib as mpl
import peakutils
import seaborn as sns
from collections import Counter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def rotateTraj(a11,theta1,c1):
	a11 = np.array(a11)
	rotMat = np.array([[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]])
	for j1 in range(a11.shape[0]):
		a11[j1,:] -= np.array(c1)
		a11[j1,:] = np.dot(rotMat,np.squeeze(a11[j1,:]))+np.array(c1)
	return a11
def add_arrow_to_line2D(
	axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
	arrowstyle='-|>', arrowsize=1, transform=None):
	"""
	Add arrows to a matplotlib.lines.Line2D at selected locations.

	Parameters:
	-----------
	axes: 
	line: Line2D object as returned by plot command
	arrow_locs: list of locations where to insert arrows, % of total length
	arrowstyle: style of the arrow
	arrowsize: size of the arrow
	transform: a matplotlib transform instance, default to data coordinates

	Returns:
	--------
	arrows: list of arrows
	"""
	if not isinstance(line, mlines.Line2D):
		raise ValueError("expected a matplotlib.lines.Line2D object")
	x, y = line.get_xdata(), line.get_ydata()

	arrow_kw = {
		"arrowstyle": arrowstyle,
		"mutation_scale": 8 * arrowsize,
	}

	color = line.get_color()
	use_multicolor_lines = isinstance(color, np.ndarray)
	if use_multicolor_lines:
		raise NotImplementedError("multicolor lines not supported")
	else:
		arrow_kw['color'] = color

	linewidth = line.get_linewidth()
	if isinstance(linewidth, np.ndarray):
		raise NotImplementedError("multiwidth lines not supported")
	else:
		arrow_kw['linewidth'] = linewidth

	if transform is None:
		transform = axes.transData

	arrows = []
	for loc in arrow_locs:
		s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
		n = np.searchsorted(s, s[-1] * loc)
		arrow_tail = (x[n], y[n])
		arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
		p = mpatches.FancyArrowPatch(
			arrow_tail, arrow_head, transform=transform,
			**arrow_kw)
		axes.add_patch(p)
		arrows.append(p)
	return arrows
def closerHorV2(x_pos,y_pos,theta,horizVector,vertVector):
	cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])+hBL*cToHead
	hVec = cVector - np.dot(cVector,horizVector)*horizVector
	vVec = cVector - np.dot(cVector,vertVector)*vertVector
	hDist = np.linalg.norm(hVec)
	vDist = np.linalg.norm(vVec)
	mDist = np.min([hDist,vDist])
	# print hDist,vDist
	#a = cVector+7.0*cToHead
	# to check geometry
	# print hDist,vDist
	#vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.
	mDist = np.min([hDist,vDist])
	if hDist<vDist:
		return 'horiz',mDist
	elif vDist<=hDist:
		return 'vert',mDist
	else:
		return 'neither',mDist
def closerHorV3(x_pos,y_pos,theta,horizVector,vertVector):
	# cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])#+hBL*cToHead
	hVec = cVector - np.dot(cVector,horizVector)*horizVector
	vVec = cVector - np.dot(cVector,vertVector)*vertVector
	hDist = np.linalg.norm(hVec)
	vDist = np.linalg.norm(vVec)
	mDist = np.min([hDist,vDist])
	# print hDist,vDist
	#a = cVector+7.0*cToHead
	# to check geometry
	# print hDist,vDist
	#vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.
	mDist = np.min([hDist,vDist])
	if hDist<vDist:
		return 'horiz',mDist
	elif vDist<=hDist:
		return 'vert',mDist

def calcQuadrant(x_pos, y_pos):

	leftRight =x_pos < y_pos*m1 +b1
	upDown = y_pos < m2*x_pos + b2

	if leftRight and not upDown:
		return 3
	elif leftRight and upDown:
		return 2
	elif not leftRight and upDown:
		return 1
	else:
		return 4

def isQuadHot(quad,state):
	if state==1:
		return quad ==2.0 or quad==4.0
	else:
		return quad ==1.0 or quad == 3.0

def isNearTempBarrier(x_pos,y_pos,theta,num,quad_hot,quad_hot_rear):
	cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])+hBL*cToHead
	hVec = cVector - np.dot(cVector,horizVector)*horizVector
	vVec = cVector - np.dot(cVector,vertVector)*vertVector
	hDist = np.linalg.norm(hVec)
	vDist = np.linalg.norm(vVec)

	mDist = np.min([hDist,vDist])
	if mDist < nearThresh_temp:

		if hDist<vDist:
			cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-hBL*cToHead
			hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector
			hdistRear = np.linalg.norm(hVec_rear)
			tVal = np.arccos(np.dot(cToHead,horizVector))*180./np.pi

			if quad_hot:
				hDist*=-1. 

			if quad_hot_rear:
				hdistRear*=-1.

			if hVec_rear[1]<0 or (hdistRear<0 and hVec_rear[1]>0):
				tVal = 180.-tVal

			if hDist > hLimit or hDist < lLimit: 
				return 0,0,hDist

			return 1,tVal,hDist

		elif vDist<=hDist:
			cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-hBL*cToHead
			vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
			vDistRear = np.linalg.norm(vVec_rear)

			tVal = np.arccos(np.dot(cToHead,vertVector))*180./np.pi

			if quad_hot:
				vDist*=-1. 

			if quad_hot_rear:
				vDistRear*=-1.

			if vVec_rear[0]>0 or (vVec_rear[0]<0 and vDistRear<0):
				tVal = 180.-tVal

			if vDist > hLimit or vDist < lLimit: 
				return 0,0,vDist

			return 1,tVal,vDist
		else:
			return 0,0,mDist
	else:
			return 0,0,mDist

# dataFname = "./output/"
inputDir = str(sys.argv[1])
if inputDir[-1] != '/':
	inputDir+='/'

colorOption = 3
behavior = []
behaviorCat = []
color = []
list1 = []
listSF = []
listFramesInTempB = []
listFNames = []
listEventNum = []
posList= []
quadList = []
maxProjList = []
angList = []
ang_val_list = []
af_list = []
ang_sum = []
vel_sum = []
tooShort = []
# scaling =3.9
# ###parameters for different temperatures at boundary
# if (inputDir.split('/')[-2]).split('_')[-1][0:2] == '40':
# 	hLimit = 12/4.7
# 	lLimit = -20/4.7
# elif (inputDir.split('/')[-2]).split('_')[-1][0:2] == '35':
# 	hLimit = 12/4.7
# 	lLimit = -20/4.7
# elif (inputDir.split('/')[-2]).split('_')[-1][0:2] == '30':
# 	hLimit = 12/4.7
# 	lLimit = -20/4.7
# else:
# 	hLimit = 12/4.7
# 	lLimit = -20/4.7


# modelFile = str(sys.argv[2])
# modelType = int(modelFile.split('.')[0].split('_')[-1])

hotQuads = [1,3] 
coldQuads = [2,4]
onW = []
onW_inC = []
k=0
nearThresh_temp = 20
nearThresh = 7
useAll = 0
maxDiff = 300
#read in all behaviors in output directory
fSeqs,allCenters,allMaxes,allStates = [],[],[],[]
vList,rvList = [],[]
scale_list = []
fig, ax = plt.subplots()
fSeqsLA,fSeqsRA,fSeqsBack = [],[],[]
# plt.gca().invert_yaxis()

# figScat,ax11 = plt.subplots()
# scat1 = ax11.scatter([],[],s=8,color='blue')
# img=None
count=0
count2=0
n=1
countAll = 0
count_move=0
allFnames = []

(yvals30,levels30,yvals35,levels35,yvals40,levels40,x2,y2,ti,ti35,
	 ti0) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")

#extract limits
eps = 0.1
inds = np.where(np.logical_and(x2>=-4-eps, x2<=4+eps))
x2_ = x2[inds]

if inputDir.split('/')[-2][0:9] == 'output_40':
	ti1 = ti0
	yvals = yvals40
	levels = levels40
elif inputDir.split('/')[-2][0:9] == 'output_30':
	ti1 = ti
	yvals = yvals30
	levels = levels30
elif inputDir.split('/')[-2][0:9] == 'output_35':
	ti1 = ti35
	yvals = yvals35
	levels = levels35
elif inputDir.split('/')[-2][0:9] == 'output_25':
	print('OUTPUT25????')
	ti1 = ti35
	yvals = yvals35
	levels = levels35
else: 
	print('ERROR')
ti1_ = np.squeeze(ti1[:,inds])

groupName=inputDir.split('/')[-3][8:]
tempName=inputDir.split('/')[-2][-2:]
if not os.path.exists('./staff/staff_plots/staff_plots_'+groupName+'/'):
	os.makedirs('./staff/staff_plots/staff_plots_'+groupName+'/')
if not os.path.exists('./staff/staff_data/staff_data_'+groupName+'/'):
	os.makedirs('./staff/staff_data/staff_data_'+groupName+'/')
#get distance at which temp threshold is reached (25.5)

threshMax = griddata(ti1[1,:],x2,25.5) 
print('threshmax=',threshMax)

justCasts = False
if justCasts:
	import matplotlib.colors as colors
	def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
		return new_cmap

	cmap = plt.get_cmap('PRGn_r')
	new_cmap = truncate_colormap(cmap, 0.15, 0.85)
	f1 = open(inputDir.split('/')[-2]+'_cast_inds.pkl','rb')
	castList =pickle.load(f1)

# new LUT color

coolwarm = cm.get_cmap('coolwarm')
newcolors = coolwarm(np.linspace(0, 1, 256))
green = np.array([0, 1., 0, 1])
newcolors[126:129, :] = green
newcmp = ListedColormap(newcolors)

print ('Processing files: ')
for filename in os.listdir(inputDir):
	if filename.split(".")[-1] == 'output':
		print (filename)
		ct=0
		(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,
			scaling,bodyLength,antennaeDist, flipQuadrants) = pickle.load(open(inputDir+filename,"rb"))
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
		hBL = 0.5*bodyLength

		###parameters for different temperatures at boundary
		hLimit = threshMax-0.5 #12/4.7*scaling
		lLimit = threshMax+5. #-20/4.7*scaling

		hLimit*=-1*scaling
		lLimit*=-1*scaling

		# vid = imageio.get_reader('/home/josh/Desktop/eP/github/flyMovement/WTs/FL50/'+filename.split('.')[0].split('/')[-1]+'.mp4',  'ffmpeg')#for demoing. 

		# computations for basic geometry of the arena
		t1 = data[3]
		t2 = data[4]

		m1 = (t1[0][1]-t1[0][0])/(t1[1][1]-t1[1][0]) # 1/slope
		m2 = (t2[1][1]-t2[1][0])/(t2[0][1]-t2[0][0])

		b1 = -m1*t1[1][1] +t1[0][1]
		b2 = -m2*t2[0][1] + t2[1][1]

		intersectCenter = np.array([(m1*b2+b1)/(1.-m1*m2),m2*((m1*b2+b1)/(1.-m1*m2))+b2])

		vertVector = np.array([(t1[0][1]-t1[0][0]),(t1[1][1]-t1[1][0])])
		vertVector = vertVector/np.linalg.norm(vertVector)

		if vertVector[1]<0:
			vertVector = -vertVector

		horizVector = np.array([(t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0])])
		horizVector = horizVector/np.linalg.norm(horizVector)
		if horizVector[0]<0:
			horizVector = -horizVector

		angVert =np.arctan2(vertVector[1],vertVector[0])
		angHoriz = np.arctan2(horizVector[1],horizVector[0])
		angVert = (180*angVert/np.pi) % 360.
		angVert = 180. - angVert # since image has inverted y axis. 
		angHoriz = 180.*angHoriz/np.pi if abs(angHoriz) < (np.pi/2) else 180.*angHoriz/np.pi -180.
		angHoriz*=-1.
		#find sequences of behaviors in which the fly is in the desired band
		# for checking arena parameters. 
		# ax11.plot(t1[0],t1[1],color='blue')
		# ax11.plot(t2[0],t2[1],color='green')
		# ax11.scatter(intersectCenter[0],intersectCenter[1],color='red')
		showQuadrants = int(filename.split(".")[0].split("_")[-1])
		if flipQuadrants == 1:
			showQuadrants = 3 - showQuadrants
		last = False
		cSeq = []
		cSeq_LA,cSeq_RA = [],[]
		cSeq_Back = []
		currentVs = []
		currentrVs = []
		for i1 in range(len(fc_x)):
			q0_head = calcQuadrant(fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1]))
			q0_rear = calcQuadrant(fc_y[i1]+hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]- hBL*np.cos(np.pi/180.*fAngs[i1]))
			hotQH = isQuadHot(q0_head,showQuadrants)
			hotQR = isQuadHot(q0_rear,showQuadrants)

			nearTout= isNearTempBarrier(fc_y[i1],fc_x[i1],fAngs[i1],i1,hotQH,hotQR)
			ax1,dist = closerHorV2(fc_y[i1],fc_x[i1],fAngs[i1],horizVector,vertVector)
			if nearTout[0]:
				cSeq.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])])
				cSeq_LA.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]) -antennaeDist*np.cos(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])-antennaeDist*np.sin(np.pi/180.*fAngs[i1])])
				cSeq_Back.append([fc_y[i1]+hBL*np.sin(np.pi/180.*fAngs[i1]) ,fc_x[i1]- hBL*np.cos(np.pi/180.*fAngs[i1])])
				cSeq_RA.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1])+antennaeDist*np.cos(np.pi/180.*fAngs[i1]) ,fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])+antennaeDist*np.sin(np.pi/180.*fAngs[i1])])
				currentVs.append(tV[i1])
				currentrVs.append(fAngs_deriv[i1])
				last=True
				lastX = fc_x[i1]
				lastY = fc_y[i1]
				ct=0
			#if end of sequence, rotate for viewing purposes and append to list. 
			elif last==True and (ct==1 or i1==(len(fc_x)-1)):
				countAll+=1
				last = False
				notUnderT = True
				j1=0
				dist1 = np.infty
				ax11 = 'neither'
				while notUnderT:
					q1 =  calcQuadrant(cSeq[j1][0],cSeq[j1][1])
					ax1,dist = closerHorV3(cSeq[j1][0],cSeq[j1][1],fAngs[j1],horizVector,vertVector)
					# print dist
					if dist<3:
						break
					else:
						if dist < dist1:
							ax11 = ax1
							dist1 = dist
							q11 = q1
						j1+=1

					if j1== len(cSeq):
						dist = dist1
						ax1 = ax11
						q1 = q11
						# print "check",dist,isQuadHot(q1,showQuadrants)
						break

				# print q1,ax1,angVert,angHoriz
				#calculate angle to rotate, and then rotate. 
				if isQuadHot(q1,showQuadrants):
					if q1==1 and ax1=='horiz':
						q1 = 4
					elif q1==1 and ax1=='vert':
						q1 = 2
					elif q1==2 and ax1=='horiz':
						q1 = 3
					elif q1==2 and ax1=='vert':
						q1 = 1
					elif q1==3 and ax1=='horiz':
						q1 = 2
					elif q1==3 and ax1=='vert':
						q1 = 4
					elif q1==4 and ax1=='horiz':
						q1 = 1
					elif q1==4 and ax1=='vert':
						q1 = 3

				# rotate head, antennae, back
				if (q1 ==1 and ax1=='horiz'):
					cSeq_rot = rotateTraj(cSeq,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,(angHoriz)/180.*np.pi,intersectCenter)

				elif (q1 ==2 and ax1=='horiz'):
					cSeq_rot = rotateTraj(cSeq,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,(angHoriz)/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,(angHoriz)/180.*np.pi,intersectCenter)

				elif q1 ==2 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,angVert/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,angVert/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,angVert/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,angVert/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='horiz':
					cSeq_rot = rotateTraj(cSeq,np.pi+angHoriz/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,np.pi+angHoriz/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,np.pi+angHoriz/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,np.pi+angHoriz/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,angVert/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,angVert/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,angVert/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,angVert/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,-(180.-angVert)/180.*np.pi,intersectCenter)
					

				elif q1 ==4 and ax1 =='horiz':
					cSeq_rot = rotateTraj(cSeq,(180.+angHoriz)/180*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,(180.+angHoriz)/180*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,(180.+angHoriz)/180*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,(180.+angHoriz)/180*np.pi,intersectCenter)

				elif q1 ==1 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotRA = rotateTraj(cSeq_RA,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotLA = rotateTraj(cSeq_LA,-(180.-angVert)/180.*np.pi,intersectCenter)
					cSeq_rotBack = rotateTraj(cSeq_Back,-(180.-angVert)/180.*np.pi,intersectCenter)

				cSeq = np.array(cSeq)
		
				wallSum = np.sum([(((cSeq[i11,0] - q4[0])**2 + (cSeq[i11,1]-q4[1])**2) > (halfWidth-nearThresh)**2) for i11 in range(cSeq.shape[0])])
				# check if entered and exited the boundary region
				threshNearBoundary = 5./4.7*scaling
				enterCheck = (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary)
				# print (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary), cSeq_rot[0,1]-intersectCenter[1], -hLimit
				exitCheck = (cSeq_rot[-1,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary) 
				exitCheck2 =  (cSeq_rot[-1,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary) 

				enterCheck_hot =  (cSeq_rot[0,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary)
				# print enterCheck,exitCheck,exitCheck2,(cSeq_rot[-1,1]-intersectCenter[1]),(hLimit-threshNearBoundary)
				# print cSeq_rot.shape[0]
				if cSeq_rot.shape[0]>5:
					moveCheck = np.sum([np.linalg.norm(cSeq_rot[j11-1,:]-cSeq_rot[j11,:]) for j11 in range(1,cSeq_rot.shape[0])])>3./4.7*scaling
				else:
					moveCheck = 0

				if moveCheck:
					count_move+=1

				# print wallSum
				# print intersectCenter,q4

				flag1 = True

				if not isQuadHot(calcQuadrant(cSeq[0][0],cSeq[0][1]),showQuadrants) and not isQuadHot(calcQuadrant(cSeq[-1][0],cSeq[-1][1]),showQuadrants)and wallSum<1 and enterCheck and exitCheck and moveCheck:
					#for turns
					allStates.append(1)
					line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0])/scaling,(cSeq_rot[:,1]-intersectCenter[1])/scaling,color='black',linewidth=0.7)
					count +=1

				elif not isQuadHot(calcQuadrant(cSeq[0][0],cSeq[0][1]),showQuadrants) and enterCheck and exitCheck and moveCheck:
					#for turns at wall
					allStates.append(2)
					count2 +=1
					line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0])/scaling,(cSeq_rot[:,1]-intersectCenter[1])/scaling,color='blue',linewidth=0.7)

				elif not isQuadHot(calcQuadrant(cSeq[0][0],cSeq[0][1]),showQuadrants) and enterCheck and exitCheck2 and moveCheck:
					#for straights not on the wall
					allStates.append(3)
					line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0])/scaling,(cSeq_rot[:,1]-intersectCenter[1])/scaling,color='black',linewidth=0.7)
					count +=1

				elif isQuadHot(calcQuadrant(cSeq[0][0],cSeq[0][1]),showQuadrants) and enterCheck_hot and exitCheck and moveCheck:
					#cross-ins
					allStates.append(4)
					count2 +=1
					line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0])/scaling,(cSeq_rot[:,1]-intersectCenter[1])/scaling,color='blue',linewidth=0.7)

				elif isQuadHot(calcQuadrant(cSeq[0][0],cSeq[0][1]),showQuadrants) and isQuadHot(calcQuadrant(cSeq[-1][0],cSeq[-1][1]),showQuadrants) and enterCheck_hot and exitCheck2 and moveCheck:
					# paradoxical
					allStates.append(5)
					count2 +=1
					line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0])/scaling,(cSeq_rot[:,1]-intersectCenter[1])/scaling,color='blue',linewidth=0.7)
			
				else: 
					# if none of the above
					flag1 = False

				if flag1:
					add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsBack.append(cSeq_rotBack)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					scale_list.append(scaling)

				cSeq = []
				cSeq_LA,cSeq_RA = [],[]
				cSeq_Back = []
				currentVs=[]
				currentrVs = []
				ct=0
			else:
				if fc_y[i1]!=0 and not np.isnan(fc_y[i1]):
					ct+=1
				# try:
				# 	if (np.abs((fc_y[i1]-lastY))+np.abs((fc_x[i1]-lastX)))<2.0:
				# 		cSeq.append([fc_y[i1]-7.0*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]+ 7.0*np.cos(np.pi/180.*fAngs[i1])])
				# 		currentVs.append(tV[i1])
				# except:
				# 	pass

# rect = Rectangle((-90,-60),180,60)
C = np.loadtxt('cmap.txt',dtype='int').astype('float')
cm1 = C/255.0
cm1 = mpl.colors.ListedColormap(cm1)
# scaling =4. 

for j1 in range(len(levels)):
	ax.plot([-90/scaling,90/scaling],[yvals[j1],yvals[j1]],color='black',linewidth=0.5)
	ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[-90/scaling,90/scaling,np.min(x2_),np.max(x2_)])

plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_every_response.svg')

#now plot them from left to right, left being the least "penetration" of the boundary, right being most. 
# newXs = np.linspace(-90,60,len(allMaxes))
plt.close()


# #get distance at which temp threshold is reached (25.5)
# threshMax = griddata(ti1[1,:],x2,25.5) 
# print 'threshmax=',threshMax
newX = 2
newX1 = 0
lineNum = 0
added=0
widths =[]

indsTurns = [j for j in range(len(allMaxes)) if (allStates[j]==1 or allStates[j]==2) and allMaxes[j]>threshMax]
indsStraights= [j for j in range(0,len(allMaxes)) if allStates[j]==3]

indsCrossIns= [j for j in range(len(allMaxes)) if allStates[j]==4]
indsParadoxical= [j for j in range(len(allMaxes)) if allStates[j]==5]

plt.clf()
sortInds = np.argsort(allMaxes)
if justCasts:
	sortIndsTurn_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if (sortInds[i1] in set(indsTurns) and (sortInds[i1] in castList))]
	sortIndsStraight_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if (sortInds[i1] in set(indsStraights) and (sortInds[i1] in castList))]
else:
	sortIndsTurn_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if sortInds[i1] in set(indsTurns)]
	sortIndsStraight_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if sortInds[i1] in set(indsStraights)]

sortIndsCrossIn_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if sortInds[i1] in set(indsCrossIns)]
sortIndsParadox_Filt = [sortInds[i1] for i1 in range(len(sortInds)) if sortInds[i1] in set(indsParadoxical)]



realCrossins = len([sortIndsCrossIn_Filt[j1] for j1 in range(len(sortIndsCrossIn_Filt)) if (allFnames[sortIndsCrossIn_Filt[j1]]==allFnames[sortIndsCrossIn_Filt[j1]-1]and sortIndsCrossIn_Filt[j1]-1 >0) or ((sortIndsCrossIn_Filt[j1]-1) in set(sortIndsParadox_Filt) and sortIndsCrossIn_Filt[j1]-1 >0)])

print ('Events in group - Turns: ', len(indsTurns),' Crosses: ',len(indsStraights))
print ('Cross Ins: ', realCrossins," Paradoxical Turns: ",len(indsParadoxical))

plt.bar(np.arange(4), [ len(indsTurns),len(indsStraights), realCrossins,len(indsParadoxical)])
plt.xticks(np.arange(4),('Turns','Crosses','Cross-Ins','Paradoxical'))
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_event_bar_chart.svg')
plt.clf()

allMaxes2 = []
allMaxes_Ind = []
allHeights = []
v_data = []
tVals_before = []
tVals_beforeRA,tVals_beforeLA = [],[]
if len(sortIndsTurn_Filt)>0:
	numRows = len(sortIndsTurn_Filt)//20+1 if len(sortIndsTurn_Filt) %20 !=0 else len(sortIndsTurn_Filt)//20
	fig,ax = plt.subplots(numRows,1)
	
	#calculate spacings between behaviors for all lines
	for i1 in range(len(sortIndsTurn_Filt)):
		sI = sortIndsTurn_Filt[i1]
		cSequence = fSeqs[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
		newX+=widthX
		added+=1
		if added >= 20:
			widths.append(newX)
			newX =0
			added =0
	widths.append(newX)

	mWidthI= np.argmax(widths)
	bigWidth = 200#widths[mWidthI]+21.*2.

	spacings =[]
	for i1 in range(len(widths)):
		spacings.append((bigWidth-(2.+widths[i1]))/20.)

	newX = 4
	added=0


	# plot in C/s
	dTdt = False
	alldTdt_turns = []

	#### first make staff figure for turns. 
	for i1 in range(len(sortIndsTurn_Filt)):
		sI = sortIndsTurn_Filt[i1]
		if np.sum(np.abs(rvList[sI]))==0.:
			continue
		cSequence = fSeqs[sI]
		cSequenceRA = fSeqsRA[sI]
		cSequenceLA = fSeqsLA[sI]
		cSequenceBack = fSeqsBack[sI]

		# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
		locI = np.argmin(cSequence[:,0])
		max1 = np.max(cSequence[:,1] - allCenters[sI][1])
		max1_arg = np.argmax(cSequence[:,1] - allCenters[sI][1])
		allMaxes2.append(max1/scale_list[sI])
		allMaxes_Ind+=[cSequence[max1_arg-3,1]- allCenters[sI][1],cSequence[max1_arg-2,1]- allCenters[sI][1],cSequence[max1_arg-1,1]- allCenters[sI][1],cSequence[max1_arg,1]- allCenters[sI][1]]
		allHeights.append(cSequence[:,1] - allCenters[sI][1])
		try:
			pIs = peakutils.indexes(np.abs(rvList[sI]),thres=0.5,min_dist=10)
			pI = np.argmin(np.abs(pIs -max1_arg))
			argRVmax = pIs[pI]
		except Exception as ex:
			# print ex
			# plt.plot(rvList[sI])
			# plt.show()
			argRVmax = np.argmax(np.abs(rvList[sI]))
		v_data.append([vList[sI][argRVmax]/scale_list[sI],rvList[sI][argRVmax]])
		locX = cSequence[locI,0]/scale_list[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
		#plot trajectory
		cSequence[:,0] = cSequence[:,0]/scale_list[sI] - locX + newX
		cSequenceBack[:,0] = cSequenceBack[:,0]/scale_list[sI] - locX + newX
		#flip x coord to match video.
		minXc = np.min(cSequence[:,0])
		maxXc = np.max(cSequence[:,0])

		cSequence[:,0] = maxXc - cSequence[:,0] + minXc
		cSequenceBack[:,0] = maxXc - cSequenceBack[:,0] + minXc

		tVals_before.append(griddata(x2,ti1[1,:],(cSequence[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))
		tVals_beforeLA.append(griddata(x2,ti1[1,:],(cSequenceLA[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))
		tVals_beforeRA.append(griddata(x2,ti1[1,:],(cSequenceRA[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))

		newX+= widthX+spacings[lineNum]
		added+=1

		ys = np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI]

		g = griddata(x2,ti1[1,:],ys)
		mix = np.gradient(g,np.arange(len(g))/30)
		alldTdt_turns.append(mix)
		if dTdt:
			pmin = -20.
			pmax = 20.
		else:
			mix = 30*np.array(np.abs(vList[sI]))/scale_list[sI]
			pmin = 0.
			pmax = 10.

		if justCasts:
			if numRows>1:
				line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				ax[lineNum].scatter(np.array(cSequence[:,0]),ys,c=rvList[sI],cmap=new_cmap,s=2,vmin=-2.5,vmax=2.5)
				ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])

			else:
				line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				ax.scatter(np.array(cSequence[:,0]),ys,c=rvList[sI],cmap=new_cmap,s=2,vmin=-2.5,vmax=2.5)
				ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
		else:
			if numRows>1:
				line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				pq=ax[lineNum].scatter(np.array(cSequence[:,0]),ys,c=mix,cmap=newcmp,s=2,vmin=pmin,vmax=pmax)
				ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				for k1 in range(0, cSequence.shape[0], 2):
					ax[lineNum].plot([cSequence[k1,0],cSequenceBack[k1,0]],[(cSequence[k1,1]-allCenters[sI][1])/scale_list[sI],(cSequenceBack[k1,1]-allCenters[sI][1])/scale_list[sI]],color='black',linewidth=0.25)
				#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])

			else:
				line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				pq=ax.scatter(np.array(cSequence[:,0]),ys,c=mix,cmap=newcmp,s=2,vmin=pmin,vmax=pmax)
				ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				for k1 in range(0, cSequence.shape[0], 2):
					ax.plot([cSequence[k1,0],cSequenceBack[k1,0]],[(cSequence[k1,1]-allCenters[sI][1])/scale_list[sI],(cSequenceBack[k1,1]-allCenters[sI][1])/scale_list[sI]],color='black',linewidth=0.25)
				#print(len(np.array(cSequence[0,0])))
				#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])

		if added >=20:
			for j1 in range(len(levels)):
				if numRows>1:
					ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				else:
					ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				tVals = griddata(x2,ti1[1,:],allMaxes2)
				tlabels = np.arange(25.5,37.5,1.0)
				sVals = griddata(ti1[1,:],x2,tlabels)
				dtVals = griddata(x2,np.gradient(ti1[1,:]),allMaxes2)
				tVals_dt = griddata(x2,ti1[1,:],allMaxes_Ind)

			if numRows>1:
				if lineNum<numRows:
					ax[lineNum].set_aspect('equal','box')
					ax[lineNum].set_anchor('W')
			else:
				ax.set_aspect('equal','box')
				ax.set_anchor('W')
			lineNum+=1
			newX1 = np.max([newX1,newX])
			newX = 2
			added=0
	if added>0:
		
		for j1 in range(len(levels)):
			if numRows>1:
				ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			else:
				ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			tVals = griddata(x2,ti[1,:],allMaxes2)
			tlabels = np.arange(25.5,37.5,1.0)
			sVals = griddata(ti[1,:],x2,tlabels)
			dtVals = griddata(x2,np.gradient(ti[1,:]),allMaxes2)
			tVals_dt = griddata(x2,ti[1,:],allMaxes_Ind)

	if numRows>1:
		if lineNum<numRows:
			ax[lineNum].set_aspect('equal','box')
			ax[lineNum].set_anchor('W')
	else:
		ax.set_aspect('equal','box')
		ax.set_anchor('W')
	newX1 = np.max([newX1,newX])
	fig.set_size_inches(12.5, 2.5*(numRows))
	# for lineNum in xrange(0,numRows):
	# 	print ax[lineNum].figure.subplotpars.hspace,ax[lineNum].figure.subplotpars.top,ax[lineNum].figure.subplotpars.bottom,ax[lineNum].get_yaxis()
	# fig.suptitle('A drosophila symphony - turns',fontsize=16)
	plt.colorbar(pq)
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_ordered_responses_turns.svg')

	'''
	# scatters the last traj
	fig,ax = plt.subplots()
	scat0 = ax.scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=30*np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=10.)
	plt.colorbar(scat0)
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+'colorbar_fig.svg')
	plt.close()
	'''

	# plot the distribution of C/s
	alldTdt_turns = np.concatenate(alldTdt_turns, axis=0)
	print(alldTdt_turns.shape)
	fig,ax = plt.subplots()
	ax = sns.violinplot(y=alldTdt_turns,bw='scott',orient='v',cut=0)
	# ax = sns.boxplot(y=alldTdt_turns,orient='v',width=0.25)
	# ax.set_ylim([0,12])
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_violin_all_dT_dt_turns.svg')
	plt.close()

	try:
		tVals_dt = np.reshape(tVals_dt,(-1,4))
		tVals_dt = np.diff(tVals_dt,axis=1)
		tVals_dt = np.mean(tVals_dt,axis=1)
	except:
		pass
	v_data = np.array(v_data)
	fig,ax = plt.subplots()
	ax.scatter(v_data[:,0],v_data[:,1])
	ax.set_xlabel('trans v')
	ax.set_ylabel('rot v')
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_boundary_trans_rot_relation.svg')
	plt.close()
	f1 = open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] + '_bdry_tr.pkl','wb')
	pickle.dump(v_data,f1)
	f1.close()
	fig,ax = plt.subplots()
	ax = sns.violinplot(y=allMaxes2,bw='scott',orient='v',cut=0)
	ax2 = plt.twinx()
	# ticks = ax.get_yticks()
	ax2.set_ylim(ax.get_ylim())
	ax2.set_yticks(sVals)
	ax2.set_yticklabels(tlabels)
	ax2.tick_params(labelsize=5)
	ax.set_ylabel('Distance from boundary(mm)')
	ax.scatter(np.zeros(len(allMaxes2)),allMaxes2,color='red',alpha=0.3)
	ax2.set_ylabel('Temperature (C)')
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_violin_dist.svg')
	plt.close()

	plt.hist(allMaxes2,bins=20)
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_hist_response.svg')
	plt.close()

	fig,ax = plt.subplots()
	ax = sns.violinplot(y=dtVals,bw='scott',orient='v')
	ax.set_ylabel('dT/dx')
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_dt_violin.svg')

	fig,ax = plt.subplots()
	ax = sns.violinplot(y=dtVals/(tVals-25.0),bw='scott',orient='v')
	ax.scatter(np.zeros(len(dtVals)),dtVals/(tVals-25.0),color='red',alpha=0.3)
	ax.set_ylabel('dT/dx/ (T-25)')
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_weber_violin.svg')


	fig,ax = plt.subplots()
	ax = sns.violinplot(y=tVals_dt,bw='scott',orient='v')
	ax.set_title(inputDir.split('/')[-2])
	ax.set_ylabel('dT/dt')
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_t_dt_violin.svg')


	fig,ax = plt.subplots()
	ax.plot(np.gradient(ti1[1,:]),'g',label='dT/dx')
	ax.plot(np.gradient(ti1[1,:])/(ti1[1,:]-25),'k',label='dT/dx/(T-25)')
	ax.legend()
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+'profile_'+inputDir.split('/')[-2][-2:]+'.svg')
	fig,ax = plt.subplots()
	ax.plot(ti1[1,:],'g',label='T')
	ax.legend()
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+'temp_profile_'+inputDir.split('/')[-2][-2:]+'.svg')

	# tVals2 = tVals[~np.isnan(tVals)]
	# print tVals[np.isnan(tVals)!=1]
	plt.hist(tVals[np.isnan(tVals)!=1],bins=20)
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_tempHist_response.svg')
	plt.close()
	with open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] +'_dists_turns.csv', mode='w') as csv_file:
		fieldnames = ['dist', 'temp']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		for i in range(len(allMaxes2)):
			writer.writerow({'dist':allMaxes2[i],'temp':tVals[i]})
	with open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] +'_tcurve.csv', mode='w') as csv_file:
		fieldnames = ['dist', 'temp']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		for i in range(len(sVals)):
			writer.writerow({'dist':sVals[i],'temp':tlabels[i]})


	list0 = list(tVals_dt)
	with open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] +'_all_preturn_dts.csv', mode='w') as csv_file:
		fieldnames = ['dT']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		for i in range(len(list0)):
			writer.writerow({'dT':list0[i]})


########## now make figure for straights
allMaxes2=[]
if len(sortIndsStraight_Filt) >0:
	numRows = len(sortIndsStraight_Filt)//20+1 if len(sortIndsStraight_Filt) %20 !=0 else len(sortIndsStraight_Filt)//20
	fig,ax = plt.subplots(numRows,1)
	newX = 2
	newX1 = 0
	lineNum = 0
	added=0
	widths =[]
	#calculate spacings between behaviors for all lines
	for i1 in range(len(sortIndsStraight_Filt)):
		sI = sortIndsStraight_Filt[i1]
		cSequence = fSeqs[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI] 
		newX+=widthX
		added+=1
		if added >=20:
			widths.append(newX)
			newX =0
			added =0
	widths.append(newX)

	mWidthI= np.argmax(widths)
	bigWidth = 200.

	spacings =[]
	for i1 in range(len(widths)):
		spacings.append((bigWidth-(2.+widths[i1]))/20.)

	# print spacings,widths
	newX = 4
	added=0

	for i1 in range(len(sortIndsStraight_Filt)):
		sI = sortIndsStraight_Filt[i1]
		cSequence = fSeqs[sI]
		# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
		locI = np.argmin(cSequence[:,0])
		max1 = np.max(cSequence[:,1] - allCenters[sI][1])/scale_list[sI]
		allMaxes2.append(max1)
		allHeights.append(cSequence[:,1] - allCenters[sI][1])
		locX = cSequence[locI,0]/scale_list[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI] 

		#plot trajectory
		cSequence[:,0] = cSequence[:,0]/scale_list[sI] - locX + newX
		newX+= widthX+spacings[lineNum]

		#flip x coord to match video.
		minXc = np.min(cSequence[:,0])
		maxXc = np.max(cSequence[:,0])

		cSequence[:,0] = maxXc - cSequence[:,0]  + minXc 
		tVals_before.append(griddata(x2,ti1[1,:],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI]))
		tVals_beforeLA.append(griddata(x2,ti1[1,:],(cSequenceLA[:,1]-allCenters[sI][1])/scale_list[sI]))
		tVals_beforeRA.append(griddata(x2,ti1[1,:],(cSequenceRA[:,1]-allCenters[sI][1])/scale_list[sI]))

		added+=1

		dTdt = False
		ys = np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI]
		g = griddata(x2,ti1[1,:],ys)
		if dTdt:
			mix = np.gradient(g,np.arange(len(g))/30)
			pmin = -20.
			pmax = 20.
		else:
			mix = 30*np.array(np.abs(vList[sI]))/scale_list[sI]
			pmin = 0.
			pmax = 10.

		if justCasts:
			if numRows>1:
				line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				ax[lineNum].scatter(np.array(cSequence[:,0]),ys,c=rvList[sI],cmap=new_cmap,s=2,vmin=-2.5,vmax=2.5)
				ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])

			else:
				line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				ax.scatter(np.array(cSequence[:,0]),ys,c=rvList[sI],cmap=new_cmap,s=2,vmin=-2.5,vmax=2.5)
				ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
		else:
			if numRows>1:
				line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				pq=ax[lineNum].scatter(np.array(cSequence[:,0]),ys,c=mix,cmap=newcmp,s=2,vmin=pmin,vmax=pmax)
				ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#for k1 in range(0, cSequence.shape[0], 4):
				#	ax[lineNum].plot([cSequence[k1,0],cSequenceBack[k1,0]],[(cSequence[k1,1]-allCenters[sI][1])/scale_list[sI],(cSequenceBack[k1,1]-allCenters[sI][1])/scale_list[sI]],color='black',linewidth=0.25)
				#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])
				#for jj in range(len(np.array(cSequence[:,0]))):
				#	ax[lineNum].plot([np.array(cSequence[:,0])[jj], np.array(cSequence[:,0])[jj]], [ys[jj], ys[jj]+1], 'k-', lw=0.1)

			else:
				line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
				pq=ax.scatter(np.array(cSequence[:,0]),ys,c=mix,cmap=newcmp,s=2,vmin=pmin,vmax=pmax)
				ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
				#for k1 in range(0, cSequence.shape[0], 4):
				#	ax.plot([cSequence[k1,0],cSequenceBack[k1,0]],[(cSequence[k1,1]-allCenters[sI][1])/scale_list[sI],(cSequenceBack[k1,1]-allCenters[sI][1])/scale_list[sI]],color='black',linewidth=0.25)
				#print(len(np.array(cSequence[0,0])))
				#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])

		if added >=20:
			for j1 in range(len(levels)):
				if numRows>1:
					ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				else:
					ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				# print x2.shape,ti0.shape,len(allMaxes)
				tlabels = np.arange(25.5,37.5,1.0)
				sVals = griddata(ti1[1,:],x2,tlabels)

			if numRows>1:
				ax[lineNum].set_aspect('equal','box')
				ax[lineNum].set_anchor('W')
			else:
				ax.set_aspect('equal','box')
				ax.set_anchor('W')
			lineNum+=1
			newX1 = np.max([newX1,newX])
			newX = 2
			added=0
	if added>0:
		for j1 in range(len(levels)):
				if numRows>1:
					ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				else:
					ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				# print x2.shape,ti0.shape,len(allMaxes)
				tlabels = np.arange(25.5,37.5,1.0)
				sVals = griddata(ti1[1,:],x2,tlabels)

	if numRows>1:
		if lineNum<numRows:
			ax[lineNum].set_aspect('equal','box')
			ax[lineNum].set_anchor('W')
	else:
		ax.set_aspect('equal','box')
		ax.set_anchor('W')
	tVals = griddata(x2,ti1[1,:],allMaxes2)
	
		# print sVals
			# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
	newX1 = np.max([newX1,newX])

	# fig.set_size_inches(5*newX1/(-np.min(x2)+np.max(x2)), 5*numRows)
	# fig.suptitle('A drosophila symphony - straights',fontsize=16)

	fig.set_size_inches(12.5, 2.5*(numRows))
	plt.colorbar(pq)
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_ordered_responses_straights.svg')
	with open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] +'_dists_straights.csv', mode='w') as csv_file:
		fieldnames = ['dist', 'temp']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		for i in range(len(allMaxes2)):
			writer.writerow({'dist':allMaxes2[i],'temp':tVals[i]})


#first remove cross-ins that happen at start of movie
sortIndsCrossIn_Filt = [sortIndsCrossIn_Filt[j1] for j1 in range(len(sortIndsCrossIn_Filt)) if (allFnames[sortIndsCrossIn_Filt[j1]]==allFnames[sortIndsCrossIn_Filt[j1]-1]and sortIndsCrossIn_Filt[j1]-1 >0)]
if len(sortIndsCrossIn_Filt)>0:
	numRows = len(sortIndsCrossIn_Filt)//20+1 if len(sortIndsCrossIn_Filt) %20 !=0 else len(sortIndsCrossIn_Filt)//20

	# numRows = len(sortIndsCrossIn_Filt)//20+1
	fig,ax = plt.subplots(numRows,1)
	newX = 2
	newX1 = 0
	lineNum = 0
	added=0
	widths =[]
	#calculate spacings between behaviors for all lines
	for i1 in range(len(sortIndsCrossIn_Filt)):
		sI = sortIndsCrossIn_Filt[i1]
		cSequence = fSeqs[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
		newX+=widthX
		added+=1
		if added >=20:
			widths.append(newX)
			newX =0
			added =0
	widths.append(newX)

	mWidthI= np.argmax(widths)
	bigWidth = 200.#widths[mWidthI]+21.*2.

	spacings =[]
	for i1 in range(len(widths)):
		spacings.append((bigWidth-(2.+widths[i1]))/20.)

	newX = 4
	added=0
	for i1 in range(len(sortIndsCrossIn_Filt)):
		sI = sortIndsCrossIn_Filt[i1]
		cSequence = fSeqs[sI]
		# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
		locI = np.argmin(cSequence[:,0])
		max1 = np.max(cSequence[:,1] - allCenters[sI][1])/scale_list[sI]
		locX = cSequence[locI,0]/scale_list[sI]
		# widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
		allHeights.append(cSequence[:,1] - allCenters[sI][1])

		#plot trajectory
		cSequence[:,0] = cSequence[:,0]/scale_list[sI]- locX + newX
		#flip x coord to match video.
		minXc = np.min(cSequence[:,0])
		maxXc = np.max(cSequence[:,0])

		cSequence[:,0] = maxXc - cSequence[:,0]  + minXc 
		newX+= widthX+spacings[lineNum]
		added+=1
		if numRows>1:
			line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
			ax[lineNum].scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=30*np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=10.)
			ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
			#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])
		else:
			line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
			ax.scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=30*np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=10.)
			ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)

			#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])

		if added >=20:
			for j1 in range(len(levels)):
				if numRows>1:
					ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				else:
					ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				# print x2.shape,ti0.shape,len(allMaxes)
				tlabels = np.arange(25.5,37.5,1.0)
				sVals = griddata(ti1[1,:],x2,tlabels)

			if numRows>1:
				if lineNum<numRows:
					ax[lineNum].set_aspect('equal','box')
					ax[lineNum].set_anchor('W')
			else:
				ax.set_aspect('equal','box')
				ax.set_anchor('W')
			lineNum+=1
			newX1 = np.max([newX1,newX])
			# print newX,newX1
			newX = 2
			added=0

	if added>0:
		for j1 in range(len(levels)):
			if numRows>1:
				ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			else:
				ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			# print x2.shape,ti0.shape,len(allMaxes)
			tlabels = np.arange(25.5,37.5,1.0)
			sVals = griddata(ti1[1,:],x2,tlabels)

		if numRows>1:
			ax[lineNum].set_aspect('equal','box')
			ax[lineNum].set_anchor('W')
		else:
			ax.set_aspect('equal','box')
			ax.set_anchor('W')

	newX1 = np.max([newX1,newX])

	# fig.suptitle('A drosophila symphony - straights',fontsize=16)

	fig.set_size_inches(12.5, 2.5*(numRows))
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_ordered_responses_crossIns.svg')

########## now make figure for paradoxical turns
if len(sortIndsParadox_Filt)>0:
	numRows = len(sortIndsParadox_Filt)//20+1 if len(sortIndsParadox_Filt) %20 !=0 else len(sortIndsParadox_Filt)//20
	fig,ax = plt.subplots(numRows,1)
	newX = 2
	newX1 = 0
	lineNum = 0
	added=0
	widths =[]
	#calculate spacings between behaviors for all lines
	for i1 in range(len(sortIndsParadox_Filt)):
		sI = sortIndsParadox_Filt[i1]
		cSequence = fSeqs[sI]
		widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
		allHeights.append(cSequence[:,1] - allCenters[sI][1])
		newX+=widthX
		added+=1
		if added >=20:
			widths.append(newX)
			newX =0
			added =0
	widths.append(newX)

	mWidthI= np.argmax(widths)
	bigWidth = 200

	spacings =[]
	for i1 in range(len(widths)):
		spacings.append((bigWidth-(2.+widths[i1]))/20.)

	newX = 4
	added=0

	for i1 in range(len(sortIndsParadox_Filt)):
		sI = sortIndsParadox_Filt[i1]
		cSequence = fSeqs[sI]
		# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
		locI = np.argmin(cSequence[:,0])
		max1 = np.max(cSequence[:,1] - allCenters[sI][1])/scale_list[sI]
		locX = cSequence[locI,0]/scale_list[sI]
		# widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]

		#plot trajectory
		cSequence[:,0] = cSequence[:,0]/scale_list[sI]- locX + newX
		#flip x coord to match video.
		minXc = np.min(cSequence[:,0])
		maxXc = np.max(cSequence[:,0])

		cSequence[:,0] = maxXc - cSequence[:,0]  + minXc 

		newX+= widthX+spacings[lineNum]
		added+=1
		if numRows>1:
			line, = ax[lineNum].plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
			ax[lineNum].scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=30*np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=10.)
			ax[lineNum].scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
			#add_arrow_to_line2D(ax[lineNum],line,arrow_locs=[0,0.5,1])
		else:
			line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
			ax.scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=30*np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=10.)
			ax.scatter(np.array(cSequence[0,0]),np.array((cSequence[0,1]-allCenters[sI][1]))/scale_list[sI],color='g',s=2)
			#add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])

		if added >=20:
			for j1 in range(len(levels)):
				if numRows>1:
					ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				else:
					ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
					ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
				# print x2.shape,ti0.shape,len(allMaxes)
				tlabels = np.arange(25.5,37.5,1.0)
				sVals = griddata(ti1[1,:],x2,tlabels)

			if numRows>1:
				if lineNum<numRows:
					ax[lineNum].set_aspect('equal','box')
					ax[lineNum].set_anchor('W')
			else:
				ax.set_aspect('equal','box')
				ax.set_anchor('W')
			lineNum+=1
			newX1 = np.max([newX1,newX])
			# print newX,newX1
			newX = 2
			added=0
	if added>0:
		for j1 in range(len(levels)):
			if numRows>1:
				ax[lineNum].plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax[lineNum].imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			else:
				ax.plot([0,newX],[yvals[j1],yvals[j1]],color='black',linewidth=0.1)
				ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
			# print x2.shape,ti0.shape,len(allMaxes)
			tlabels = np.arange(25.5,37.5,1.0)
			sVals = griddata(ti1[1,:],x2,tlabels)

	if numRows>1:
		ax[lineNum].set_aspect('equal','box')
		ax[lineNum].set_anchor('W')
	else:
		ax.set_aspect('equal','box')
		ax.set_anchor('W')
		# print sVals
			# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
	newX1 = np.max([newX1,newX])

	# fig.set_size_inches(5*newX1/(-np.min(x2)+np.max(x2)), 5*numRows)
	# fig.suptitle('A drosophila symphony - straights',fontsize=16)

	fig.set_size_inches(12.5, 2.5*(numRows))
	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_ordered_responses_paradox.svg')


###################################################################################
#saving data for later processing

nVs = [vList[sortIndsTurn_Filt[j1]] for j1 in range(len(sortIndsTurn_Filt))]
nrVs = [rvList[sortIndsTurn_Filt[j1]]for j1 in range(len(sortIndsTurn_Filt))]
nDs = [fSeqs[sortIndsTurn_Filt[j1]][:,1] -allCenters[sortIndsTurn_Filt[j1]][1] for j1 in range(len(sortIndsTurn_Filt))]
scales = [scale_list[sortIndsTurn_Filt[j1]] for j1 in range(len(sortIndsTurn_Filt))]

nTs = [griddata(x2,ti1[1,:],nDs[j1]) for j1 in range(len(sortIndsTurn_Filt))]
if inputDir.split('/')[-2][0:9] == 'output_25':
	nTs = False


if nTs:
	fout = 'staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] + '_turning_data.pkl'
	w1 = open(fout,"wb")
	datOut = (nVs,nrVs,nDs,nTs,scales)
	pickle.dump(datOut,w1)
	w1.close()

nVs = [vList[sortIndsStraight_Filt[j1]] for j1 in range(0,len(sortIndsStraight_Filt))]
nrVs = [rvList[sortIndsStraight_Filt[j1]]for j1 in range(0,len(sortIndsStraight_Filt))]
scales = [scale_list[sortIndsStraight_Filt[j1]]for j1 in range(0,len(sortIndsStraight_Filt))]
nDs = [fSeqs[sortIndsStraight_Filt[j1]][:,1] -allCenters[sortIndsStraight_Filt[j1]][1] for j1 in range(0,len(sortIndsStraight_Filt))]

nTs = [griddata(x2,ti1[1,:],nDs[j1]) for j1 in range(0,len(sortIndsStraight_Filt))]
if inputDir.split('/')[-2][0:9] == 'output_25':
	nTs = False

if nTs:
	fout = 'staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] + '_straights_data.pkl'
	w1 = open(fout,"wb")
	datOut = (nVs,nrVs,nDs,nTs,scales)
	pickle.dump(datOut,w1)
	w1.close()
realCrossins = len([sortIndsCrossIn_Filt[j1] for j1 in range(0,len(sortIndsCrossIn_Filt)) if (allFnames[sortIndsCrossIn_Filt[j1]]==allFnames[sortIndsCrossIn_Filt[j1]-1]and sortIndsCrossIn_Filt[j1]-1 >0) ])

### export data for making a box plot
turnsInVids = Counter([allFnames[j] for j in range(0,len(allStates)) if (allStates[j]==1 or allStates[j]==2)])
straightsInVids = Counter([allFnames[j]for j in range(0,len(allStates)) if (allStates[j]==3)])
crossInsInVids = Counter([allFnames[j] for j in range(0,len(allStates)) if (allStates[j]==4 )])
paradoxTurnsInVids = Counter([allFnames[j] for j in range(0,len(allStates)) if (allStates[j]==5)])


fout = 'staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] + '_boxplot_data.pkl'
w1 = open(fout,"wb")
datOut = (turnsInVids,straightsInVids,crossInsInVids,paradoxTurnsInVids,allFnames)
pickle.dump(datOut,w1)
w1.close()

# # now make same plot, but with rotational information. 
# plt.close()
# plt.close()
# fig,ax = plt.subplots()
# newX = 2
# for i1 in xrange(0,len(allMaxes)):
# 	sI = sortInds[i1]
# 	cSequence = fSeqs[sI]
# 	# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
# 	locI = np.argmin(cSequence[:,0])
# 	max1 = np.max(cSequence[:,1] - allCenters[sI][1])
# 	locX = cSequence[locI,0]
# 	widthX = np.max(cSequence[:,0]) - np.min(cSequence[:,0])


# 	if cSequence.shape[0]>3: #and allStates[sI]==1:
# 		cSequence[:,0] = cSequence[:,0]- locX + newX
# 		newX+= widthX+2.

# 		line, = ax.plot(cSequence[:,0]/scale_list[sI],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
# 		print cSequence.shape,len(rvList[sI])
# 		ax.scatter(np.array(cSequence[:,0])/scale_list[sI],np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=np.abs(rvList[sI]),cmap=plt.cm.cool,s=2,vmin=0,vmax=5.)
# 		add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])


# if inputDir.split('/')[-2] == 'output_40FL50':
# 	for j1 in xrange(0,len(levels40)):
# 		ax.plot([0,newX],[yvals40[j1],yvals40[j1]],color='black',linewidth=0.1)
# 		ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
# 		# print x2.shape,ti0.shape,len(allMaxes)
# 		tVals = griddata(x2,ti0[1,:],allMaxes)
# 		tlabels = np.arange(25.5,37.5,1.0)
# 		sVals = griddata(ti0[1,:],x2,tlabels)


# if inputDir.split('/')[-2] == 'output_30FL50':
# 	for j1 in xrange(0,len(levels30)):
# 		ax.plot([0,newX],[yvals30[j1],yvals30[j1]],color='black',linewidth=0.5)
# 		ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
# 		tVals = griddata(x2,ti[1,:],allMaxes)
# 		tlabels = np.arange(25.5,29.5,1.0)
# 		sVals = griddata(ti[1,:],x2,tlabels)

# if inputDir.split('/')[-2] == 'output_35FL50':
# 	for j1 in xrange(0,len(levels35)):
# 		ax.plot([0,newX],[yvals35[j1],yvals35[j1]],color='black',linewidth=0.5)
# 		ax.imshow(ti1_.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2_),np.max(x2_)])
# 		tVals = griddata(x2,ti35[1,:],allMaxes)
# 		tlabels = np.arange(25.5,33.5,1.0)
# 		sVals = griddata(ti35[1,:],x2,tlabels)


# ax.set_aspect('equal','box')
# fig.set_size_inches(5*newX/(-np.min(x2)+np.max(x2)), 5)
# plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_ordered_responses_rotational.svg')

##make dT/dt plots

dtVals_before = [np.gradient(tVals_before[i]) for i in range(0,len(tVals_before)) if len(tVals_before[i])>2]
diff_tVals_before = [tVals_beforeRA[i]-tVals_beforeLA[i] for i in range(0,len(tVals_beforeLA)) if len(tVals_beforeLA[i])>2]
#max plots
maxes_before = [30*np.max(dtVals_before[i]) for i in range(0,len(dtVals_before))]
fig,ax = plt.subplots()
ax = sns.violinplot(y=maxes_before,orient='v')
ax.set_title(inputDir.split('/')[-2])
ax.set_ylabel('dT/dt')
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_max_dT_dt_violin.svg')
#average plots
means_before = [30*np.mean(dtVals_before[i]) for i in range(0,len(dtVals_before))]
fig,ax = plt.subplots()
ax = sns.violinplot(y=means_before,orient='v')
ax.set_title(inputDir.split('/')[-2])
ax.set_ylabel('dT/dt')
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_mean_dT_dt_violin.svg')



fig,ax = plt.subplots()
ax = sns.boxplot(y=means_before,orient='v',width=0.25)
ax = sns.swarmplot(y=means_before,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_means_box_dT_dt.svg')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(y=maxes_before,orient='v',width=0.25)
ax = sns.swarmplot(y=maxes_before,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_maxes_box_dT_dt.svg')
plt.close()


maxes_before2 = [np.argmax(np.abs(diff_tVals_before[i])) for i in range(0,len(diff_tVals_before))]
maxes_before2 = [diff_tVals_before[i][maxes_before2[i]] for i in range(0,len(maxes_before2))]
means_before2 = [np.mean(diff_tVals_before[i]) for i in range(0,len(diff_tVals_before))]

fig,ax = plt.subplots()
ax = sns.boxplot(y=means_before2,orient='v',width=0.25)
ax = sns.swarmplot(y=means_before2,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_means_box_differenceT_dt.svg')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(y=maxes_before2,orient='v',width=0.25)
ax = sns.swarmplot(y=maxes_before2,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_maxes_box_differenceT_dt.svg')
plt.close()


maxes_before3 = [np.argmax(np.abs(np.gradient(diff_tVals_before[i]))) for i in range(0,len(diff_tVals_before))]
maxes_before3 = [30*diff_tVals_before[i][maxes_before3[i]] for i in range(0,len(maxes_before3))]
means_before3 = [30*np.mean(np.gradient(diff_tVals_before[i])) for i in range(0,len(diff_tVals_before))]

fig,ax = plt.subplots()
ax = sns.boxplot(y=means_before3,orient='v',width=0.25)
ax = sns.swarmplot(y=means_before3,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_means_box_rate_differenceT_dt.svg')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(y=maxes_before3,orient='v',width=0.25)
ax = sns.swarmplot(y=maxes_before3,color=".25",orient='v')
# ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_maxes_box_rate_differenceT_dt.svg')
plt.close()


f1 = open('staff/staff_data/staff_data_'+groupName+'/'+inputDir.split('/')[-2] + '_dT_info.pkl','wb')
dat1 = (maxes_before,means_before,maxes_before2,means_before2,maxes_before3,means_before3)
pickle.dump(dat1,f1)
f1.close()


# #plotting all dTs in boundary region
# if 0:
# 	if inputDir.split('/')[-2][0:9] == 'output_40':
# 		tVals_dt = [list(np.diff(griddata(x2,ti0[1,:],allHeights[i]))) for i in xrange(0,len(allHeights))]


# 	if inputDir.split('/')[-2][0:9] == 'output_30':
# 		tVals_dt = [list(np.diff(griddata(x2,ti[1,:],allHeights[i]))) for i in xrange(0,len(allHeights))]

# 	if inputDir.split('/')[-2][0:9] == 'output_35':
# 		tVals_dt = [list(np.diff(griddata(x2,ti35[1,:],allHeights[i]))) for i in xrange(0,len(allHeights))]

# 	from itertools import chain
# 	fig,ax = plt.subplots()
# 	ax = sns.violinplot(list(chain.from_iterable(tVals_dt)),bw='silverman',orient='v')
# 	ax.set_title(inputDir.split('/')[-2])
# 	ax.set_ylabel('dT/dt')
# 	plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/'+inputDir.split('/')[-2] + '_all_dt_violin.svg')

# 	#write out to csv format. 
# 	list0 = list(chain.from_iterable(tVals_dt))
# 	with open(inputDir.split('/')[-2] +'_all_dts.csv', mode='w') as csv_file:
# 		fieldnames = ['dT']
# 		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

# 		writer.writeheader()
# 		for i in xrange(0,len(list0)):
# 			writer.writerow({'dT':list0[i]})




############ make fancy histograms here now ###############
# maxTsTurns = []
# for i in xrange(0,len(nTs)):
# 	if np.max(nTs[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])>25.5:
# 		maxTsTurns.append(nDs[i][nbeforeListT[i]+np.argmax(nTs[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])])


	########################################################################
