import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
#from plot_boundary_trajs import *
import os
import io
from PIL import Image
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon,ttest_ind
from scipy.stats import mannwhitneyu,ks_2samp


def rotateTraj(a11,theta1,c1):
	a11 = np.array(a11)
	rotMat = np.array([[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]])
	for j1 in range(0,a11.shape[0]):
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
	#a = cVector+hBL*cToHead
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
	#a = cVector+hBL*cToHead
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
posList= []
quadList = []
maxProjList = []
angList = []
ang_val_list = []
af_list = []
ang_sum = []
vel_sum = []
tooShort = []

# ###parameters for different temperatures at boundary
# if (inputDir.split('/')[-2]).split('_')[-1] == '40FL50':
# 	hLimit = 12
# 	lLimit = -20
# elif (inputDir.split('/')[-2]).split('_')[-1] == '35FL50':
# 	hLimit = 12
# 	lLimit = -20
# elif (inputDir.split('/')[-2]).split('_')[-1] == '30FL50':
# 	hLimit = 12
# 	lLimit = -20
# else:
# 	hLimit = 12
# 	lLimit = -20


# modelFile = str(sys.argv[2])
# modelType = int(modelFile.split('.')[0].split('_')[-1])

hotQuads = [1,3] 
coldQuads = [2,4]
onW = []
onW_inC = []
k=0
import imageio
nearThresh_temp = 20
nearThresh = 7
useAll = 0
maxDiff = 300
#read in all behaviors in output directory
fSeqs,allCenters,allMaxes,allStates = [],[],[],[]
fSeqsLA,fSeqsRA = [],[]
fSeqsCentroid = []
vList,rvList = [],[]
nbeforeList,nafterList = [],[]
start_time_list =[]
# fig, ax = plt.subplots()
# plt.gca().invert_yaxis()

# figScat,ax11 = plt.subplots()
# scat1 = ax11.scatter([],[],s=8,color='blue')
# img=None

from scipy.interpolate import griddata
import pickle
(yvals30,levels30,yvals35,levels35,yvals40,levels40,x2,y2,ti,ti35,
	 ti0) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")

if inputDir.split('/')[-2][0:9] == 'output_40' or inputDir.split('/')[-2][0:9] == 'output_25' :
	col1 = 'red'
	yvals =yvals40
	levels = levels40
	ti1 = ti0
if inputDir.split('/')[-2][0:9] == 'output_30':
	col1 = 'yellow'
	yvals =yvals30
	levels = levels30
	ti1 = ti
if inputDir.split('/')[-2][0:9] == 'output_35':
	col1 = 'orange'
	yvals =yvals35
	levels = levels35
	ti1 = ti35

groupName=inputDir.split('/')[-3][8:]
tempName=inputDir.split('/')[-2][-2:]
if not os.path.exists('./bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'):
	os.makedirs('./bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/')
if not os.path.exists('./bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'):
	os.makedirs('./bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/')
#get distance at which temp threshold is reached (25.5)

#get distance at which temp threshold is reached (25.5)
threshMax = griddata(ti1[1,:],x2,25.5) 
print ('threshmax=',threshMax)

count=0
n=1
count2=0
countAll = 0
count_move=0
allFnames = []
scale_list =[]
for filename in os.listdir(inputDir):
	print(filename)
	if filename.split(".")[-1] == 'output':
		ct=0
		(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,scaling,bodyLength,antennaeDist, flipQuadrants) = pickle.load(open(inputDir+filename,"rb"))
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
		# vid = imageio.get_reader('/home/josh/Desktop/eP/github/flyMovement/WTs/FL50/'+filename.split('.')[0].split('/')[-1]+'.mp4',  'ffmpeg')#for demoing. 


		###parameters for different temperatures at boundary
		hLimit = threshMax-0.5
		lLimit = threshMax+5.

		hLimit*=-1*scaling
		lLimit*=-1*scaling


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
		angHoriz = 180.*angHoriz/np.pi if abs(angHoriz) < (np.pi/2) else 180.*angHoriz/np.pi -180.
		angVert = 180. - angVert 
		angHoriz*=-1.

		#find sequences of behaviors in which the fly is in the desired band
		# ax11.plot(t1[0],t1[1],color='blue')
		# ax11.plot(t2[0],t2[1],color='green')
		# ax11.scatter(intersectCenter[0],intersectCenter[1],color='red')
		showQuadrants = int(filename.split(".")[0].split("_")[-1])
		if flipQuadrants == 1:
			showQuadrants = 3 - showQuadrants
		last = False
		cSeq,cSeq_centroid = [],[]
		cSeq_LA,cSeq_RA = [],[]
		currentVs = []
		currentrVs = []
		for i1 in range(0,len(fc_x)):
			q0_head = calcQuadrant(fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1]))
			q0_rear = calcQuadrant(fc_y[i1]+hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]- hBL*np.cos(np.pi/180.*fAngs[i1]))
			hotQH = isQuadHot(q0_head,showQuadrants)
			hotQR = isQuadHot(q0_rear,showQuadrants)
			# print q0_head,q0_rear,hotQH,hotQR
			nearTout= isNearTempBarrier(fc_y[i1],fc_x[i1],fAngs[i1],i1,hotQH,hotQR)
			ax1,dist = closerHorV2(fc_y[i1],fc_x[i1],fAngs[i1],horizVector,vertVector)
			# print ax1,dist
			# im5 = vid.get_data(i1)
			# if img is None:
			# 	img = ax11.imshow(im5)
			# else:
			# 	img.set_data(im5)
			# scat1.set_offsets(np.c_[fc_y[0:i1+1]-7*np.sin(np.pi/180.*fAngs[0:i1+1]),fc_x[0:i1+1]+7*np.cos(np.pi/180*fAngs[0:i1+1])])
			# plt.pause(0.0001)

			# plt.draw()
			if nearTout[0] and last==False:
			
				# iMax = np.max([i1,0])
				maxBack = 1
				iMax = i1 - maxBack
				i10 = i1
				lastX = fc_x[i1]
				lastY = fc_y[i1]
				n1 = 0
				nBefore=0

				for i0 in reversed(range(iMax,i1+1)):
					if np.abs(lastX - fc_x[i0])<5 and np.abs(lastY - fc_y[i0])<5:
						cSeq.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])])
						cSeq_LA.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) -antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])-antennaeDist*np.sin(np.pi/180.*fAngs[i0])])
						cSeq_RA.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) +antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])+antennaeDist*np.sin(np.pi/180.*fAngs[i0])])
						cSeq_centroid.append([fc_y[i0],fc_x[i0]])
						currentVs.append(tV[i0])
						currentrVs.append(fAngs_deriv[i0])
						lastX = fc_x[i0]
						lastY = fc_y[i0]						
						nBefore = i1-i0
					else:
						# nBefore = i1-i0-1
						# print lastX,fc_x[i0],'x',i1-i0
						# print lastY,fc_y[i0],'y'
						# plt.plot(fc_y)
						# plt.show()
						# cSeq,cSeq_centroid,cSeq_LA,cSeq_RA,currentrVs,currentVs = [],[],[],[],[],[]
						break
				cSeq = list(reversed(cSeq))
				cSeq_centroid = list(reversed(cSeq_centroid))
				cSeq_LA = list(reversed(cSeq_LA))
				cSeq_RA = list(reversed(cSeq_RA))
				currentVs = list(reversed(currentVs))
				currentrVs = list(reversed(currentrVs))
				current_start_time = iMax
				ct=0
				last=True
				lastX = fc_x[i1]
				lastY = fc_y[i1]
				# if len(cSeq)<1:
				# 	print nBefore
				# 	nBefore = 0
				# 	asdf


			elif nearTout[0] and last==True:
		
				#append to sequence
				cSeq.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])])
				cSeq_centroid.append([fc_y[i1],fc_x[i1]])
				cSeq_LA.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]) -antennaeDist*np.cos(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])-antennaeDist*np.sin(np.pi/180.*fAngs[i1])])
				cSeq_RA.append([fc_y[i1]-hBL*np.sin(np.pi/180.*fAngs[i1]) +antennaeDist*np.cos(np.pi/180.*fAngs[i1]),fc_x[i1]+ hBL*np.cos(np.pi/180.*fAngs[i1])+antennaeDist*np.sin(np.pi/180.*fAngs[i1])])
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
				j1=nBefore#np.min([5,len()

				# let's also tack on a few frames for behaviors that continue following exit of the region
				# if nafter <30:
				# print nafter, 'nafter'
				# print lastX,lastY,fc_x[i0],fc_y[i0],fc_x[i0-1],fc_y[i0-1]

				i1_loc = i1 - ct
				nafter =np.min([1,len(fc_x)-i1_loc-1])

				for i0 in range(i1_loc,i1_loc+nafter):
					if np.abs(lastX - fc_x[i0])<5 and np.abs(lastY - fc_y[i0])<5:
						cSeq.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])])
						cSeq_LA.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) -antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])-antennaeDist*np.sin(np.pi/180.*fAngs[i0])])
						cSeq_RA.append([fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) +antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])+antennaeDist*np.sin(np.pi/180.*fAngs[i0])])
						cSeq_centroid.append([fc_y[i0],fc_x[i0]])
						currentVs.append(tV[i0])
						currentrVs.append(fAngs_deriv[i0])
						lastX = fc_x[i0]
						lastY = fc_y[i0]
						nafter = i0- i1_loc+1
					else:
						# print  np.abs(lastX - fc_x[i0]),np.abs(lastY - fc_y[i0]),fc_x[i0-1],lastX
						nafter = i0- i1_loc
						break
				dist1 = np.infty
				ax11 = 'neither'
				# print len(cSeq),nafter,nBefore
				while notUnderT:
					q1 =  calcQuadrant(cSeq[j1][0],cSeq[j1][1])
					ax1,dist = closerHorV3(cSeq[j1][0],cSeq[j1][1],fAngs[j1],horizVector,vertVector)
					# print dist
					if dist<5:
						break
					else:
						if dist < dist1:
							ax11 = ax1
							dist1 = dist
						j1+=1

					if j1== len(cSeq) - nafter:
						ax1 = ax11
						dist = dist1
						# print "check",dist,isQuadHot(q1,showQuadrants)
						break
				# print q1,ax1,angVert,angHoriz
				#calculate angle to rotate, and then rotate. 
				# print q1,ax1
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
				# print q1

				if (q1 ==1 and ax1=='horiz'):
					cSeq_rot = rotateTraj(cSeq,(angHoriz)/180.*np.pi,intersectCenter)
				elif (q1 ==2 and ax1=='horiz'):
					cSeq_rot = rotateTraj(cSeq,(angHoriz)/180.*np.pi,intersectCenter)

				elif q1 ==2 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,angVert/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='horiz':
					cSeq_rot = rotateTraj(cSeq,np.pi+angHoriz/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,angVert/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,-(180.-angVert)/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='horiz':
					cSeq_rot = rotateTraj(cSeq,(180.+angHoriz)/180*np.pi,intersectCenter)

				elif q1 ==1 and ax1 =='vert':
					cSeq_rot = rotateTraj(cSeq,-(180.-angVert)/180.*np.pi,intersectCenter)
				# same for each of the antennae 

				if (q1 ==1 and ax1=='horiz'):
					cSeq_rotRA = rotateTraj(cSeq_RA,(angHoriz)/180.*np.pi,intersectCenter)

				elif (q1 ==2 and ax1=='horiz'):
					cSeq_rotRA = rotateTraj(cSeq_RA,(angHoriz)/180.*np.pi,intersectCenter)

				elif q1 ==2 and ax1 =='vert':
					cSeq_rotRA = rotateTraj(cSeq_RA,angVert/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='horiz':
					cSeq_rotRA = rotateTraj(cSeq_RA,np.pi+angHoriz/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='vert':
					cSeq_rotRA = rotateTraj(cSeq_RA,angVert/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='vert':
					cSeq_rotRA = rotateTraj(cSeq_RA,-(180.-angVert)/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='horiz':
					cSeq_rotRA = rotateTraj(cSeq_RA,(180.+angHoriz)/180*np.pi,intersectCenter)

				elif q1 ==1 and ax1 =='vert':
					cSeq_rotRA = rotateTraj(cSeq_RA,-(180.-angVert)/180.*np.pi,intersectCenter)

				#left antenna
				if (q1 ==1 and ax1=='horiz'):
					cSeq_rotLA = rotateTraj(cSeq_LA,(angHoriz)/180.*np.pi,intersectCenter)

				elif (q1 ==2 and ax1=='horiz'):
					cSeq_rotLA = rotateTraj(cSeq_LA,(angHoriz)/180.*np.pi,intersectCenter)

				elif q1 ==2 and ax1 =='vert':
					cSeq_rotLA = rotateTraj(cSeq_LA,angVert/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='horiz':
					cSeq_rotLA = rotateTraj(cSeq_LA,np.pi+angHoriz/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='vert':
					cSeq_rotLA = rotateTraj(cSeq_LA,angVert/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='vert':
					cSeq_rotLA = rotateTraj(cSeq_LA,-(180.-angVert)/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='horiz':
					cSeq_rotLA = rotateTraj(cSeq_LA,(180.+angHoriz)/180*np.pi,intersectCenter)

				elif q1 ==1 and ax1 =='vert':
					cSeq_rotLA = rotateTraj(cSeq_LA,-(180.-angVert)/180.*np.pi,intersectCenter)


				#centroid
				if (q1 ==1 and ax1=='horiz'):
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,(angHoriz)/180.*np.pi,intersectCenter)

				elif (q1 ==2 and ax1=='horiz'):
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,(angHoriz)/180.*np.pi,intersectCenter)

				elif q1 ==2 and ax1 =='vert':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,angVert/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='horiz':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,np.pi+angHoriz/180.*np.pi,intersectCenter)

				elif q1 ==3 and ax1 =='vert':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,angVert/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='vert':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,-(180.-angVert)/180.*np.pi,intersectCenter)

				elif q1 ==4 and ax1 =='horiz':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,(180.+angHoriz)/180*np.pi,intersectCenter)

				elif q1 ==1 and ax1 =='vert':
					cSeq_rotCentroid = rotateTraj(cSeq_centroid,-(180.-angVert)/180.*np.pi,intersectCenter)

				cSeq = np.array(cSeq)

				# if q1==3 and ax1=='vert':

				# 	plt.plot(cSeq_rot[:,0],cSeq_rot[:,1]-intersectCenter[1])
				# 	plt.plot(cSeq[:,0],cSeq[:,1],color='red')
				# 	plt.plot(cSeq[-1,0],cSeq[-1,1],'*',color='black')

				# 	plt.plot(cSeq_rot[-1,0],cSeq_rot[-1,1]-intersectCenter[1],'*',color ='green')
				# 	plt.gca().invert_yaxis()
				# 	print cSeq_rot[:,1] - intersectCenter[1]
				# 	plt.show()


				# cSeq = []
				# fSeqs.append(cSeq_rot)
				# print cSeq_rot.shape
				# cSeq = np.array(cSeq)
				# cSeq_LA = np.array(cSeq_LA)
				# cSeq_RA = np.array(cSeq_RA)
				# check if on wall
				# if n==15:
				# 	print cSeq_rot[nBefore:len(cSeq_rot)-nafter]
				# 	print (cSeq_rot.shape[0]-nBefore-nafter)>5,np.sum([np.linalg.norm(cSeq_rot[j11-1,:]-cSeq_rot[j11,:]) for j11 in range(nBefore+1, cSeq_rot.shape[0]-nafter)])>3.
				# 	print (cSeq_rot[nBefore,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary)
				# 	# print (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary), cSeq_rot[0,1]-intersectCenter[1], -hLimit
				# 	print (cSeq_rot[-nafter-1,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary) 
				# 	print (cSeq_rot[-nafter-1,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary) 

				# 	print (cSeq_rot[nBefore,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary)
				# 	asdf
				# else:
				# 	n+=1
				wallSum = np.sum([(((cSeq[i11,0] - q4[0])**2 + (cSeq[i11,1]-q4[1])**2) > (halfWidth-nearThresh)**2) for i11 in range(nBefore,cSeq_rot.shape[0]-nafter)])
				# check if entered and exited the boundary region
				threshNearBoundary_hot = 5*scaling/4.7
				threshNearBoundary_cold = 5*scaling/4.7
				# enterCheck = np.any([(cSeq_rot[j0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary) for j0 in range(0,dist0)])
				# # enterCheck = (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary)
				# # print (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary), cSeq_rot[0,1]-intersectCenter[1], -hLimit
				# exitCheck = np.any([(cSeq_rot[-j0-1,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary) for j0 in range(0,nafter)])
				# exitCheck2 =  np.any([(cSeq_rot[-j0-1,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary) for j0 in range(0,nafter)])
				# enterCheck_hot =   np.any([(cSeq_rot[0,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary)for j0 in range(0,dist0)])
				# print cSeq_rot[-nafter-1]
				# asdfasd
				enterCheck = (cSeq_rot[nBefore,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary_cold)
				# print (cSeq_rot[0,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary), cSeq_rot[0,1]-intersectCenter[1], -hLimit
				exitCheck = (cSeq_rot[-nafter-1,1]-intersectCenter[1]) < -(hLimit-threshNearBoundary_cold) 
				exitCheck2 =  (cSeq_rot[-nafter-1,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary_hot) 

				enterCheck_hot =  (cSeq_rot[nBefore,1]-intersectCenter[1]) > -(lLimit+threshNearBoundary_hot)
				if (cSeq_rot.shape[0]-nBefore-nafter)>5:
					moveCheck = np.sum([np.linalg.norm(cSeq_rot[j11-1,:]-cSeq_rot[j11,:]) for j11 in range(nBefore+1, cSeq_rot.shape[0]-nafter)])>3*scaling/4.7
				else:
					moveCheck = 0

				if moveCheck:
					count_move+=1
				# print wallSum
				# print intersectCenter,q4
				#for turns
				if not isQuadHot(calcQuadrant(cSeq[nBefore][0],cSeq[nBefore][1]),showQuadrants) and not isQuadHot(calcQuadrant(cSeq[-nafter-1][0],cSeq[-nafter-1][1]),showQuadrants)and wallSum<1 and enterCheck and exitCheck and moveCheck:
					# fig,ax = plt.subplots()
					# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
					# line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0]),(cSeq_rot[:,1]-intersectCenter[1]),color='black',linewidth=0.7)
					# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
					count +=1
					# plt.gca().invert_yaxis()
					# plt.show()
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsCentroid.append(cSeq_rotCentroid)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allStates.append(1)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					nafterList.append(nafter)
					nbeforeList.append(nBefore)
					start_time_list.append(current_start_time)
					scale_list.append(scaling)

				#for turns at wall
				elif not isQuadHot(calcQuadrant(cSeq[nBefore][0],cSeq[nBefore][1]),showQuadrants) and enterCheck and exitCheck and moveCheck:
					# fig,ax = plt.subplots()
					# line, = ax.plot((cSeq_rot[:,0] -intersectCenter[0]),(cSeq_rot[:,1]-intersectCenter[1]),color='black',linewidth=0.7)
					# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])

					count2 +=1

					# plt.gca().invert_yaxis()
					# plt.show()
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsCentroid.append(cSeq_rotCentroid)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allStates.append(2)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					nafterList.append(nafter)
					nbeforeList.append(nBefore)
					start_time_list.append(current_start_time)
					scale_list.append(scaling)


					#for straights not on the wall
				elif not isQuadHot(calcQuadrant(cSeq[nBefore][0],cSeq[nBefore][1]),showQuadrants) and enterCheck and exitCheck2 and moveCheck:
					# fig,ax = plt.subplots()
					# line, = ax.plot((cSeq[:,0] -intersectCenter[0]),(cSeq[:,1]-intersectCenter[1]),color='grey',linewidth=0.7)
					# add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
					count +=1
					# plt.gca().invert_yaxis()
					# plt.show()
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsCentroid.append(cSeq_rotCentroid)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allStates.append(3)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					nafterList.append(nafter)
					nbeforeList.append(nBefore)
					start_time_list.append(current_start_time)
					scale_list.append(scaling)


				elif isQuadHot(calcQuadrant(cSeq[nBefore][0],cSeq[nBefore][1]),showQuadrants) and enterCheck_hot and exitCheck and moveCheck:
					#cross-ins
					count2 +=1
					# plt.gca().invert_yaxis()
					# plt.show()
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsCentroid.append(cSeq_rotCentroid)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allStates.append(4)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					nafterList.append(nafter)
					nbeforeList.append(nBefore)
					start_time_list.append(current_start_time)
					scale_list.append(scaling)


				elif isQuadHot(calcQuadrant(cSeq[nBefore][0],cSeq[nBefore][1]),showQuadrants) and isQuadHot(calcQuadrant(cSeq[-nafter-1][0],cSeq[-nafter-1][1]),showQuadrants) and enterCheck_hot and exitCheck2 and moveCheck:
					# paradoxical
					count2 +=1
					# plt.gca().invert_yaxis()
					# plt.show()
					fSeqs.append(cSeq_rot)
					fSeqsLA.append(cSeq_rotLA)
					fSeqsRA.append(cSeq_rotRA)
					fSeqsCentroid.append(cSeq_rotCentroid)
					allCenters.append(intersectCenter)
					allMaxes.append(np.max(cSeq_rot[:,1]-intersectCenter[1])/scaling)
					allStates.append(5)
					allFnames.append(filename)
					vList.append(currentVs)
					rvList.append(currentrVs)
					nafterList.append(nafter)
					nbeforeList.append(nBefore)
					start_time_list.append(current_start_time)
					scale_list.append(scaling)



				cSeq = []
				cSeq_LA,cSeq_RA = [],[]
				cSeq_centroid = []
				currentVs=[]
				currentrVs = []
				
			else:
				if fc_y[i1]!=0 and not np.isnan(fc_y[i1]):
					ct+=1
from scipy.interpolate import griddata

indsTurns = [j for j in range(0,len(allMaxes)) if (allStates[j]==1 or allStates[j]==2) and allMaxes[j]>threshMax]
indsStraights= [j for j in range(0,len(allMaxes)) if allStates[j]==3]
indsCrossIns= [j for j in range(0,len(allMaxes)) if allStates[j]==4]
indsParadoxical= [j for j in range(0,len(allMaxes)) if allStates[j]==5]


sortInds = np.argsort(allMaxes)
sortIndsTurn_Filt = [sortInds[i1] for i1 in range(0,len(sortInds)) if sortInds[i1] in set(indsTurns)]
sortIndsStraights_Filt = [sortInds[i1] for i1 in range(0,len(sortInds)) if sortInds[i1] in set(indsStraights)]
sortIndsCrossIn_Filt = [sortInds[i1] for i1 in range(0,len(sortInds)) if sortInds[i1] in set(indsCrossIns)]
sortIndsParadox_Filt = [sortInds[i1] for i1 in range(0,len(sortInds)) if sortInds[i1] in set(indsParadoxical)]

from collections import Counter
turnsInVids = Counter([allFnames[j] for j in range(0,len(allStates)) if (allStates[j]==1 or allStates[j]==2) and allMaxes[j]>threshMax])
straightsInVids = Counter([allFnames[j]for j in range(0,len(allStates)) if (allStates[j]==3)])
crossInsInVids = Counter([allFnames[j] for j in range(1,len(allStates)) if (allStates[j]==4 and (allFnames[j]==allFnames[j-1]))])
paradoxTurnsInVids = Counter([allFnames[j] for j in range(0,len(allStates)) if (allStates[j]==5)])

fout = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] + '_boxplot_data.pkl'
w1 = open(fout,"wb")
datOut = (turnsInVids,straightsInVids,crossInsInVids,paradoxTurnsInVids,allFnames)
pickle.dump(datOut,w1)
w1.close()

#### store turns/straights along with the corresponding interaction number for stratified testing (CMH)
normalInt = set([1,2,3])
int_num = []
ct1 = Counter()
for j in range(0,len(allStates)):
	int_num.extend([ct1[allFnames[j]]])
	ct1[allFnames[j]]+=1


### calculate the effective time (starting with first interaction with the heat)

times_turns_straights = [start_time_list[j] for j in range(0,len(start_time_list)) if  allStates[j] in normalInt and allMaxes[j]>threshMax]
filenames_turns_straights = [allFnames[j] for j in range(0,len(start_time_list)) if  allStates[j] in normalInt and allMaxes[j]>threshMax]
turns_straights = [(allStates[j]==1 or allStates[j]==2) for j in range(0,len(start_time_list)) if  allStates[j] in normalInt and allMaxes[j]>threshMax]

c00 = Counter()
d00 = dict()
for j in range(0,len(times_turns_straights)):
	if c00[filenames_turns_straights[j]] ==0:
		d00.update({filenames_turns_straights[j] : times_turns_straights[j]})
#

times_turns_straights_adj = [times_turns_straights[j] - d00[filenames_turns_straights[j]] for j in range(0,len(times_turns_straights))]
interaction_list = [(int_num[j],(allStates[j]==1 or allStates[j]==2)) for j in range(0,len(allStates)) if allStates[j] in normalInt and allMaxes[j]>threshMax]
#### save filename/time info for later analysis. 
fout = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] + '_stratified_ratio_data.pkl'
w1 = open(fout,"wb")
pickle.dump(interaction_list,w1)
w1.close()

### store turns/straights, but now with the fly name (i.e. organized video name)
interaction_list0 = [(filenames_turns_straights[j],turns_straights[j],times_turns_straights_adj[j]) for j in range(0,len(turns_straights))]

# interaction_list0 = [(allFnames[j],(allStates[j]==1 or allStates[j]==2)) for j in range(0,len(allStates)) if allStates[j] in normalInt and allMaxes[j]>threshMax]

fout = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] + '_byVideo_ratio_data.pkl'
w1 = open(fout,"wb")
pickle.dump(interaction_list0,w1)
w1.close()

###

nVs = [np.array(vList[sortIndsTurn_Filt[j1]])*30/scale_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nrVs = [rvList[sortIndsTurn_Filt[j1]]for j1 in range(0,len(sortIndsTurn_Filt))]
nDs = [(fSeqs[sortIndsTurn_Filt[j1]][:,1] -allCenters[sortIndsTurn_Filt[j1]][1])/scale_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nDsName = [allFnames[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))] # for Jenna, to label maxTs
nDs_RA = [(fSeqsRA[sortIndsTurn_Filt[j1]][:,1] -allCenters[sortIndsTurn_Filt[j1]][1])/scale_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nDs_LA = [(fSeqsLA[sortIndsTurn_Filt[j1]][:,1] -allCenters[sortIndsTurn_Filt[j1]][1])/scale_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nDs_centroid = [(fSeqsCentroid[sortIndsTurn_Filt[j1]][:,1] -allCenters[sortIndsTurn_Filt[j1]][1])/scale_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nbeforeListT = [nbeforeList[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
nafterListT = [nafterList[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]
# startListT = [start_time_list[sortIndsTurn_Filt[j1]] for j1 in range(0,len(sortIndsTurn_Filt))]

### get data for check 2
indsTurns = [j for j in range(0,len(allMaxes)) if (allStates[j]==1 or allStates[j]==2) and allMaxes[j]>threshMax]
indsTurnsAndStraights = [j for j in range(0,len(allMaxes)) if (allStates[j]==1 or allStates[j]==2 or allStates[j]==3) and allMaxes[j]>threshMax]
indsStraights= [j for j in range(0,len(allMaxes)) if allStates[j]==3]

indsTurns = indsTurnsAndStraights
nDs2 = [(fSeqs[indsTurns[j1]][:,1] -allCenters[indsTurns[j1]][1])/scale_list[indsTurns[j1]] for j1 in range(0,len(indsTurns)) ]
nFnames = [allFnames[ indsTurns[j1]] for j1 in range(0,len(indsTurns)) ]
startListT2 = [start_time_list[indsTurns[j1]] for j1 in range(0,len(indsTurns))]
nVs2 = [np.array(vList[indsTurns[j1]])*30/scale_list[indsTurns[j1]] for j1 in range(0,len(indsTurns))]
nRVs2 = [rvList[indsTurns[j1]]for j1 in range(0,len(indsTurns))]

######

if inputDir.split('/')[-2][0:9] == 'output_40' or inputDir.split('/')[-2][0:9] == 'output_25' :
	nTs = [griddata(x2,ti0[1,:],nDs[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_RA = [griddata(x2,ti0[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_LA = [griddata(x2,ti0[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_centroid = [griddata(x2,ti0[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs2 = [griddata(x2,ti0[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]


elif inputDir.split('/')[-2][0:9] == 'output_30':
	nTs = [griddata(x2,ti[1,:],nDs[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_RA = [griddata(x2,ti[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_LA = [griddata(x2,ti[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_centroid = [griddata(x2,ti[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs2 = [griddata(x2,ti[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]

elif inputDir.split('/')[-2][0:9] == 'output_35':
	nTs = [griddata(x2,ti35[1,:],nDs[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_RA = [griddata(x2,ti35[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_LA = [griddata(x2,ti35[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_centroid = [griddata(x2,ti35[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs2 = [griddata(x2,ti35[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]


elif inputDir.split('/')[-2][0:9] == 'output_25':
	nTs = [griddata(x2,ti0[1,:],nDs[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_RA = [griddata(x2,ti0[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_LA = [griddata(x2,ti0[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]
	nTs_centroid = [griddata(x2,ti0[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsTurn_Filt))]


else:
	nTs = False


# if nTs:
# 	fout = inputDir.split('/')[-2] + '_boundary_tData.pkl'
# 	w1 = open(fout,"wb")
# 	datOut = (nVs,nrVs,nDs,nTs,nTs_RA,nTs_LA,nTs_centroid,nbeforeListT,nafterListT)
# 	pickle.dump(datOut,w1)
# 	w1.close()


#####
# int_num_T_S = [ interaction_list[si][0] for si in range(0,len(interaction_list))]
# first_bin_or_not = [(1 if (np.max(nTs2[si])>25.5 and np.max(nTs2[si])<=26.5) else 0) for si in range(0,len(interaction_list))]

# fout = inputDir.split('/')[-2] + '_stratified_early_turn_ratio_data.pkl'
# w1 = open(fout,"wb")
# pickle.dump((int_num_T_S,first_bin_or_not),w1)
# w1.close()
## now organize by fly name for mixed model regression

times_turns_straights_adj2 = [start_time_list[j] - d00[allFnames[indsTurns[j]]] for j in range(0,len(indsTurns))]

interaction_list0 = [(allFnames[indsTurns[j]],(1 if (np.max(nTs2[j])>25.5 and np.max(nTs2[j])<=26.5) else 0),times_turns_straights_adj2[j]) for j in range(0,len(indsTurns))]
fout = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] + '_byVideo_early_turn_data.pkl'
w1 = open(fout,"wb")
pickle.dump(interaction_list0,w1)
w1.close()


#putting stuff together for histograms. 
maxTsTurns = []
maxTsNames = []
maxTsTurns_1,maxTsTurns_Diff=[],[]
for i in range(0,len(nTs)):
	if np.max(nTs[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])>25.5:
		maxTsTurns.append(nDs[i][nbeforeListT[i]+np.argmax(nTs[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])])
		maxTsNames.append(nDsName[i])
		#now do max temp at either antenna
		lInd = np.argmax(nTs_LA[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])
		rInd = np.argmax(nTs_RA[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]])
		if nTs_LA[i][lInd]> nTs_RA[i][rInd]:
			maxTsTurns_1.append(nDs_LA[i][lInd])
			hInd = lInd
		else:
			maxTsTurns_1.append(nDs_RA[i][rInd])
			hInd = rInd
		print (hInd)
		#difference in temps at the antennae
		thing1 =np.abs(np.array(nTs_LA[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]]) - np.array(nTs_RA[i][nbeforeListT[i]:len(nTs[i])-nafterListT[i]]))
		try:
			maxTsTurns_Diff.append(np.max(thing1[0:hInd]))
		except:
			maxTsTurns_Diff.append(thing1[hInd])


#average speed during 25.5-26 bin 
# v1s = []
# for i1 in range(0,len(nTs)):
# 	if np.max(nTs[i1]) >25.5 and np.max(nTs[i1])<=26.5:
# 		inds1 = [nTs[i1][j1]>=25.5 and nTs[i1][j1]<=26.5 for j1 in range(0,len(nTs[i1]))]
# 		# print inds1
# 		if np.sum(inds1)>0:
# 			v1s.append(np.mean(np.array(nVs[i1])[inds1]))
# fig,ax = plt.subplots()
# ax.hist(v1s,density=True, stacked=True)
# ax.set_xlabel('speed in 25.5-26.5 box')
# ax.set_ylabel('fraction')
# plt.savefig(inputDir.split('/')[-2] +'_vel_try.svg')

#just average incoming speed in 25.5-26.5 bin
v1s = []
v1s_times = []
for i1 in range(0,len(nTs2)):
	mT = np.nanmax(nTs2[i1])
	if mT >25.5:
		mT_i = np.nanargmax(nTs2[i1])
		inds1 = []
		madeIt = 0
		for j1 in range(0,mT_i):
			cond = nTs2[i1][j1]>=25.5 and nTs2[i1][j1]<=26.5
			if cond:
				inds1+=[j1]
			if madeIt==1 and not cond:
				break
			# inds1 = nTs[i1][j1]>=25.5 and nTs[i1][j1]<=26.5 for j1 in range(0,len(nTs[i1]))]
		# print inds1
		if np.sum(inds1)>0:
			v1s.append(np.mean(np.array(nVs2[i1])[inds1]))
			v1s_times.append(startListT2[i1]+inds1[0])
fig,ax = plt.subplots()
ax.hist(v1s,bins=list(np.linspace(0,20,15)),density=True, stacked=True)
ax.set_xlabel('incoming speed in 25.5-26.5 box')
ax.set_ylabel('fraction')
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_vel_try.svg')

# #incoming velocity vs time
# fig,ax = plt.subplots()
# ax.set_ylabel('incoming speed in 25.5-26.5 box')
# ax.set_xlabel('time')
# plt.savefig(inputDir.split('/')[-2] +'_vel_try.svg')


# asdfa
###commented out for now... fix later
# ## take behaviors from the same video. check if there are trends for temp experienced between subsequent interactions. 
taver = [[] for i in range(16)]
taver_times = [[] for i in range(16)]
t1s_fnames = []
t1s = []
t1s_times = []
t1s_vs,t1s_rvs = [],[]
currTs = []
currTs_times = []
currTs_start_times,currTs_end_times = [],[]
t1s_end_times,t1s_start_times = [],[]
cFname = nFnames[0]
n=0
print (len(nTs2))
### bootstrap by resampling the temperatures experienced
def get_average_incoming_vel(vProfile,tProfile,startTime):
	mT = np.nanmax(tProfile)
	mT_i = np.nanargmax(tProfile)
	inds1 = []
	madeIt = 0
	for j1 in range(0,mT_i+1):
		cond = tProfile[j1]>=25.5 and tProfile[j1]<=26.5
		if cond:
			inds1+=[j1]
		if madeIt==1 and not cond:
			break
	return np.mean(np.array(vProfile)[inds1])#,startTime+inds1[0]

currVs = []
currRVs = []

# perm  = np.random.randint(0,len(nTs2),len(nTs2))
# nTs2 = [nTs2[p] for p in perm]
for i1 in range(0,len(nTs2)):
	if nFnames[i1]== cFname:
		currTs.append(np.nanmax(nTs2[i1]))
		currTs_times.append(np.nanargmax(nTs2[i1])+startListT2[i1])
		currTs_start_times.append(startListT2[i1])
		currTs_end_times.append(startListT2[i1]+len(nTs2[i1]))
		currVs.append(get_average_incoming_vel(nVs2[i1],nTs2[i1],startListT2[i1]))
		currRVs.append(np.nanmax(np.abs(nRVs2[i1])))
	else:
		if len(currTs)>0 and np.sum(np.isnan(currTs))<1:
			# if len(currTs)>1:

			# 	# #special quick hack

			# 	# currTs = [np.max(nTs2[i1])]
			# 	# currTs_times = [np.argmax(nTs2[i1])+startListT2[i1]]
			# 	# currTs_start_times = [startListT2[i1]]
			# 	# currTs_end_times = [startListT2[i1]+ len(nTs2[i1])]
			# 	# cFname = nFnames[i1]
			# 	# continue
			# 	n+=1
			# 	stillUnder = 1
			# 	j1 = 0
			# 	if (currTs_times[j1]>0 and currTs[j1]>=currTs[j1+1]):
			# 		j1 = 0
			# 	elif (currTs_times[j1+1]>0 and currTs[j1]<currTs[j1+1]):
			# 		j1 = 1
			# 		currTs = currTs[j1:len(currTs)]
			# 		currTs_times = currTs_times[j1:len(currTs_times)]
			# 		currTs_start_times = currTs_start_times[j1:len(currTs_start_times)]
			# 		currTs_end_times = currTs_end_times[j1:len(currTs_end_times)]
				# while stillUnder:
				# 	if (currTs_times[j1]>300 and currTs[j1]>currTs[j1+1]):
				# 		stillUnder=0
				# 	elif j1==(len(currTs)-1):
				# 		stillUnder = 0
				# 		j1 = 0
				# 	else:
				# 		j1+=1
				# currTs = currTs[j1:len(currTs)]
				# currTs_times = currTs_times[j1:len(currTs_times)]
				# currTs_start_times = currTs_start_times[j1:len(currTs_start_times)]
				# currTs_end_times = currTs_end_times[j1:len(currTs_end_times)]
			# if len(currTs_times)>1:
			# 	print np.max(np.diff(currTs_times))
			# 	if np.max(np.diff(currTs_times))<1200:

			t1s.append(currTs)
			t1s_times.append(currTs_times)
			t1s_start_times.append(currTs_start_times)
			t1s_end_times.append(currTs_end_times)
			t1s_fnames.append(cFname)
			t1s_vs.append(currVs)
			t1s_rvs.append(currRVs)
			print (len(currTs),nFnames[i1-1])
			for j1 in range(0,len(currTs)):
				if len(currTs)>5:
					if len(taver)<len(currTs):
						taver.append([])
						# taver_times.append([])
					taver[j1].append(currTs[j1])
					# else:
					# 	taver[j1].append(np.nan)
						# taver_times[j1].append(currTs_times[j1])
		currTs = [np.nanmax(nTs2[i1])]
		currTs_times = [np.nanargmax(nTs2[i1])+startListT2[i1]]
		currTs_start_times = [startListT2[i1]]
		currTs_end_times = [startListT2[i1]+ len(nTs2[i1])]
		currRVs = [np.nanmax(np.abs(nRVs2[i1]))]
		currVs = [get_average_incoming_vel(nVs2[i1],nTs2[i1],startListT2[i1])]
		cFname = nFnames[i1]
box_ts= [[],[],[],[],[],[],[]]
for i in range(0,len(t1s)):
	# if np.sum(np.isnan(t1s[i])) <1:
	plt.plot(t1s[i])
	for j in range(0,np.min([len(t1s[i]),7])):
		box_ts[j].append(t1s[i][j])

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_interaction_maxtemp_plot.svg')
plt.close()


import csv
with open('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] +'_turning_temps_registered.csv', mode='w') as sensory_file:
	writer = csv.writer(sensory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)   
	for i in range(0,len(t1s)):
		writer.writerow(list(np.array(t1s_times[i])-t1s_times[i][0]))
		writer.writerow(t1s[i])
with open('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] +'_turning_temps.csv', mode='w') as sensory_file:
	writer = csv.writer(sensory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)   
	for i in range(0,len(t1s)):
		writer.writerow(t1s_times[i])
		writer.writerow(t1s[i])


early = []
late = []
cut01,cut02 = 900,900
cut03 = 1800
# maxD = 600
allTimes, allNs,allTemps= [],[],[]
allVs = []
fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	if len(t1s[i])>=0:# and np.max(np.array(t1s_times[i]) - t1s_times[i][0])<2700:
	# if np.sum(np.isnan(t1s[i])) <1:
		ax.plot((np.array(t1s_times[i]) - t1s_times[i][0])/30.,t1s[i])
		for j in range(0,len(t1s[i])):
			if (t1s_times[i][j] - t1s_times[i][0])<3600:
				allTimes.append((t1s_times[i][j] - t1s_times[i][0])/30.)
				allNs.append(j)
				allTemps.append(t1s[i][j])
				allVs.append(t1s_vs[i][j])
				if (t1s_times[i][j] - t1s_times[i][0])<cut01:
					early.append(t1s[i][j])
				elif (t1s_times[i][j] - t1s_times[i][0])>=cut02 and (t1s_times[i][j] - t1s_times[i][0])<cut03:
					late.append(t1s[i][j])

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_interaction_maxtemp_plot2.svg')
plt.close()

import seaborn as sns
print ('30_60',mannwhitneyu(early,late), len(early),len(late))
fig,ax = plt.subplots()
xdata,ydata = [],[]
xdata = [0]*len(early)+[1]*len(late)
ydata = early+late
xdata = np.array(xdata)
ydata = np.array(ydata)
sns.boxplot(x=xdata,y=ydata)
sns.swarmplot(x=xdata,y=ydata,color='black')
ax.set_ylim([25,39])
ax.set_xticklabels(['0-30 seconds','30-60 seconds'])
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_30_60_window.svg')
plt.close()
# asdfa


fout = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] + '_learning_data.pkl'
w1 = open(fout,"wb")
datOut = (t1s_times,t1s,t1s_fnames)
pickle.dump(datOut,w1)
w1.close()


fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	if len(t1s[i])>=3:# and np.sum(np.array(t1s_vs[i])<-10)<1.:
	# if np.sum(np.isnan(t1s[i])) <1:
		ax.scatter(np.array(t1s_times[i]) - t1s_times[i][0],t1s_vs[i],alpha = 0.4)

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_interaction_vel_plot.svg')
plt.close()

fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	if len(t1s[i])>=3:# and np.sum(np.array(t1s_vs[i])<-10)<1.:
	# if np.sum(np.isnan(t1s[i])) <1:
		ax.scatter(np.array(t1s_times[i]) - t1s_times[i][0],t1s_rvs[i],alpha = 0.4)

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_interaction_rot_vel_plot.svg')
plt.close()


fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	if len(t1s[i])>=3:
		ax.scatter(t1s[i],t1s_rvs[i],alpha = 0.4)
ax.set_ylabel('max ang vel')
ax.set_xlabel('max temp')
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_rot_vel_vs_max_temp.svg')
plt.close()
### run an ols on the data
import pandas as pd
from statsmodels.formula.api import ols
import seaborn as sns

df1 = pd.DataFrame(data={'Times':allTimes,'Temp':allTemps,'N':allNs,'Velocity':allVs})
# df1 = pd.DataFrame(data={'Times':[0]*len(early) +[1]*len(late),'Temp':early+late})
formula = 'Temp ~ Times'
lm = ols(formula, df1).fit()
print (lm.summary())
# print mannwhitneyu(early,late)
fig,ax = plt.subplots()

sns.regplot(x='Times',y = 'Temp',data=df1)
for i in range(0,len(t1s)):
	if len(t1s[i])>1:
	# if np.sum(np.isnan(t1s[i])) <1:
		t0 = np.array(t1s_times[i]) - t1s_times[i][0]
		t0_i = [j for j in range(0,len(t0)) if (t1s_times[i][j] - t1s_times[i][0])<3600 ]
		t0_i = np.max(t0_i)
		ax.plot((np.array(t1s_times[i][0:t0_i+1]) - t1s_times[i][0])/30.,t1s[i][0:t0_i+1])
# ax.set_ylim([25,39])
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'thingy.svg')

plt.close()
fig,ax = plt.subplots()
sns.regplot(x='Times',y = 'Velocity',data=df1)
for i in range(0,len(t1s)):
	if len(t1s[i])>=0:
	# if np.sum(np.isnan(t1s[i])) <1:
		t0 = np.array(t1s_times[i]) - t1s_times[i][0]
		t0_i = [j for j in range(0,len(t0)) if (t1s_times[i][j] - t1s_times[i][0])<2700 ]
		t0_i = np.max(t0_i)
		ax.plot(np.array(t1s_times[i][0:t0_i+1]) - t1s_times[i][0],t1s_vs[i][0:t0_i+1])
		print (t1s_vs[i][0:t0_i+1])
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_vel_thingy.svg')
# aasdf

##plotting all behaviors
fig,ax = plt.subplots()
xdata,ydata = [],[]
for i in range(0,len(box_ts)):
	xdata = xdata+[i]*len(box_ts[i])
	ydata = ydata+box_ts[i]
xdata = np.array(xdata)
ydata = np.array(ydata)
sns.boxplot(x=xdata,y=ydata)
# sns.swarmplot(x=xdata,y=ydata,color='k')
for i in range(0,len(t1s)):
	l1 = np.min([len(box_ts),len(t1s[i])])
	ax.plot(range(0,l1),t1s[i][0:l1],color='grey',alpha=0.4)
	ax.scatter(range(0,l1),t1s[i][0:l1],facecolors='none', edgecolors='k')
ax.set_ylim([24,40])
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_all_interactions_boxes.svg')
plt.close()

### perform paired wilcoxon test between 1,2 and 2,3


# thresh1 = 27
# #this isn't right, but is what one would do if the data wasn't on a bounded domain
# g1 = [t1s[i][0] for i in range(0,len(t1s)) if len(t1s[i])>1 and t1s[i][0]>thresh1]
# g2 = [t1s[i][1] for i in range(0,len(t1s)) if len(t1s[i])>1 and t1s[i][0]>thresh1]
# print wilcoxon(g1,g2)
# # g1 = [t1s[i][0] for i in range(0,len(t1s)) if len(t1s[i])>2]
# g2 = [t1s[i][1] for i in range(0,len(t1s)) if len(t1s[i])>2 and t1s[i][1]>thresh1]
# g3 = [t1s[i][2] for i in range(0,len(t1s)) if len(t1s[i])>2 and t1s[i][1]>thresh1]
# print wilcoxon(g2,g3)
try:
	print (mannwhitneyu(box_ts[0],box_ts[1],alternative='greater'))
	print (mannwhitneyu(box_ts[0],box_ts[2],alternative='greater'))
	print (mannwhitneyu(box_ts[0],box_ts[5],alternative='greater'))
except ValueError as e:
	print('MANN WHITENEY NOT WORKING!!!')
	print(e)

print (ks_2samp(box_ts[0],box_ts[1]))

# compute mean temp following crossover/deep, compare to that of shallow
afterShallow,afterDeep = [],[]
threshold = 28.
nmax=1000
for i in range(0,len(t1s)):
	for j in range(0,np.min([len(t1s[i]),nmax])):
		if j==0:
			afterShallow.append(t1s[i][j])
		elif t1s[i][j-1]>threshold:
			afterDeep.append(t1s[i][j])
		elif t1s[i][j-1]<threshold:
			afterShallow.append(t1s[i][j])
fig,ax = plt.subplots()

x1 = np.array(len(afterShallow)*[0] + len(afterDeep)*[1])
y1 = np.array(afterShallow+afterDeep)
sns.boxplot(x=x1,y=y1)
sns.swarmplot(x=x1,y=y1,color='k')
print (np.mean(afterShallow),np.mean(afterDeep))
ax.set_ylim([24,40])
ax.set_xticklabels(['After Shallow','After Deep'])
ax.set_ylabel('Max Temperature Experienced')
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'check_after.svg')
plt.close()
# print ttest_ind(afterShallow,afterDeep)
print (mannwhitneyu(afterShallow,afterDeep,alternative='greater'))
# compute mean temp before crossover/deep, compare to that of shallow
beforeShallow,beforeDeep = [],[]
threshold = 28.
for i in range(0,len(t1s)):
	for j in range(0,len(t1s[i])-1):
		if t1s[i][j+1]>threshold:
			beforeDeep.append(t1s[i][j])
		elif t1s[i][j+1]<threshold:
			beforeShallow.append(t1s[i][j])
fig,ax = plt.subplots()

x1 = np.array(len(beforeShallow)*[0] + len(beforeDeep)*[1])
y1 = np.array(beforeShallow+beforeDeep)
sns.boxplot(x=x1,y=y1)
print (np.mean(beforeShallow),np.mean(beforeDeep))
sns.swarmplot(x=x1,y=y1,color='k')
ax.set_ylim([24,40])
ax.set_xticklabels(['Before Shallow','Before Deep'])
ax.set_ylabel('Max Temperature Experienced')
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'check_before.svg')
plt.close()

# print ttest_ind(beforeShallow,beforeDeep)
print (mannwhitneyu(beforeShallow,beforeDeep,alternative='greater'))

# ### calculate conditional probability table
# #let's binarize about a threshold probability
# threshold = 28. 
# high_lows = []
# for i in range(0,len(t1s)):
# 	high_lows.append(np.array(t1s[i])>threshold)


# # if high T, prob came from low or no exposure
# HL,LL,LH,HH = 0,0,0,0
# for i in range(0,len(high_lows)):
# 	for j in range(0,len(high_lows[i])):
# 		if j==0:
# 			if high_lows[i][j]==1:
# 				HL+=1
# 			else:
# 				LL+=1
# 		else:
# 			if high_lows[i][j]==1 and high_lows[i][j-1]==0:
# 				HL+=1
# 			elif high_lows[i][j]==0 and high_lows[i][j-1]==0:
# 				LL+=1
# 			elif high_lows[i][j]==1 and high_lows[i][j-1]==1:
# 				HH+=1
# 			elif high_lows[i][j]==0 and high_lows[i][j-1]==1:
# 				LH+=1

# print HL,LL,LH,HH

# asdf
# #if low T, prob came from low or no exposure


#if high T, prob came from high exposure

#if low T, prob came from high exposure

# if 1:
# 	#act as if sampling from random distribution, uniform in space 
# 	num_resamples = 10000
# 	nTot = 0
# 	deviation= 0
# 	thresh1 = 27.
# 	for i in range(0,len(t1s)):
# 		if len(t1s[i])>1:
# 			if t1s[i][0]>thresh1:
# 				nTot += 1
# 				deviation+=t1s[i][1]-t1s[i][0]
# 		# if len(t1s[i])>2:
# 		# 	if t1s[i][1]>thresh1:
# 		# 		nTot += 1
# 		# 		deviation+=t1s[i][2]-t1s[i][1]

# 	### form an empirical null
# 	num_seqs = nTot
# 	nDevs = []
# 	for i in range(0,num_resamples):
# 		dists_rand = np.random.uniform(threshMax-0.5,threshMax+5.0,len(nTs2)*2*6)
# 		rand_maxes = [griddata(x2,ti1[1,:],dists_rand) for j1 in range(0,len(indsTurns))]
# 		#choose first 2 turns, n times according to the rule above. 
# 		#turn 1:
# 		j1=0
# 		turn1_list= []
# 		turn2_list = []
# 		for j in range(0,num_seqs):
# 			stillUnder = 1
# 			while stillUnder:
# 				turn1 = rand_maxes[j]
# 				if turn1>thresh1:
# 					stillUnder=0
# 				else:
# 					j1+=1
# 			turn1_list.append(turn1)
# 			turn2_list.append(rand_maxes[j])
# 			j1+=2
# 		# print turn1_list,turn2_list
# 		#calculate number of ups/downs
# 		nDevs.append(np.sum([turn2_list[jj]-turn1_list[jj] for jj in range(0,len(turn1_list))]))
# 	sumGreater = np.sum([n>deviation for n in nDevs])
# 	print 1.-sumGreater/(1.0*len(nDevs))

### form an empirical null
if 0:
	num_resamples = 10000
	nTot = 0
	thresh1 = 27.
	for i in range(0,len(t1s)):
		if len(t1s[i])>1:
			if t1s[i][0]>thresh1:
				nTot += 1

		# if len(t1s[i])>2:
		# 	if t1s[i][1]>thresh1:
		# 		nTot+=1
		# if len(t1s[i])>3:
		# 	if t1s[i][2]>thresh1:
		# 		nTot+=1
		# if len(t1s[i])>4:
		# 	if t1s[i][3]>thresh1:
		# 		nTot+=1
	print (nTot, np.sum([len(t1s[i])>1 for i in range(0,len(t1s))])+np.sum([len(t1s[i])>2 for i in range(0,len(t1s))]))
	num_seqs = nTot #np.sum([len(t1s[i])>1 for i in range(0,len(t1s))])+np.sum([len(t1s[i])>2 for i in range(0,len(t1s))])
	nUps = []
	for i in range(0,num_resamples):
		perm  = np.random.randint(0,len(nTs2),2*len(nTs2))
		nTsNull = [nTs2[p] for p in perm]
		#choose first 2 turns, n times according to the rule above. 
		#turn 1:
		j1=0
		turn1_list= []
		turn2_list = []
		for j in range(0,num_seqs):
			stillUnder = 1
			while stillUnder:
				turn1 = np.nanmax(nTsNull[j1])
				if turn1>thresh1:
					stillUnder=0
				else:
					j1+=1
			turn1_list.append(turn1)
			turn2_list.append(np.nanmax(nTsNull[j1+1]))
			j1+=2
		# print turn1_list,turn2_list
		#calculate number of ups/downs
		nUps.append(np.sum([turn1_list[jj]<=(turn2_list[jj]+0.1) for jj in range(0,len(turn1_list))]))

	# print len(turn2_list)
	# print nUps
	n,bins,p1s = plt.hist(nUps,bins=range(0,num_seqs+1),density=True, stacked=True)
	#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_empirical_null.svg')
	plt.close()

	nUps = 0
	#calculate how many ups we actually have:
	for i in range(0,len(t1s)):
		if len(t1s[i])>1:
			if t1s[i][0]>thresh1:
				nUps += t1s[i][0]<=(t1s[i][1]+0.1)

		if len(t1s[i])>2:
			if t1s[i][1]>thresh1:
				nUps += t1s[i][1]<=(t1s[i][2]+0.1)

		# if len(t1s[i])>3:
		# 	if t1s[i][2]>thresh1:
		# 		nUps += t1s[i][2]<=(t1s[i][3])
		# if len(t1s[i])>4:
		# 	if t1s[i][3]>thresh1:
		# 		nUps += t1s[i][3]<=(t1s[i][4])
	print (np.sum(n[0:nUps+1]),nUps)

if 0:
	##let's try to do a similar test, but now we want to calculate the deviations following a deep. 
	num_resamples = 1000
	nTot = 0
	deviation= 0
	thresh1 = 30.
	maxDist = 100
	for i in range(0,len(t1s)):
		for j in range(0,np.min([maxDist,len(t1s[i])])):
			if len(t1s[i])>j+1:
				if t1s[i][j]>thresh1:
					nTot += 1
					deviation+=t1s[i][j+1]-t1s[i][j]

	### form an empirical null
	num_seqs = nTot
	nDevs = []
	a,b =threshMax-0.5,threshMax+5.
	for i in range(0,num_resamples):
		perm  = np.random.randint(0,len(nTs2),2*len(nTs2))
		nTsNull = [nTs2[p] for p in perm]

		#choose first 2 turns, n times according to the rule above. 
		#turn 1:
		j1=0
		turn1_list= []
		turn2_list = []
		for j in range(0,num_seqs):
			stillUnder = 1
			while stillUnder:
				turn1 =np.nanmax(nTsNull[j1])# griddata(x2,ti1[1,:],np.random.uniform(a,b))#np.nanmax(nTsNull[j1])
				if turn1>thresh1:
					stillUnder=0
				else:
					j1+=1
			turn1_list.append(turn1)
			turn2_list.append(np.nanmax(nTsNull[j1+1]))#griddata(x2,ti1[1,:],np.random.uniform(a,b)))#np.nanmax(nTsNull[j1+1]))
			j1+=2
		# print turn1_list,turn2_list
		#calculate number of ups/downs
		nDevs.append(np.sum([turn2_list[jj]-turn1_list[jj] for jj in range(0,len(turn1_list))]))
	# print nDevs
	# print len(turn2_list)
	# print nUps
	# n,bins,p1s = plt.hist(nDevs,bins=100,density=True, stacked=True)
	# plt.savefig(inputDir.split('/')[-2] +'_empirical_null2.svg')
	# plt.close()
	sumGreater = np.sum([n>deviation for n in nDevs])
	print (1.-sumGreater/(1.0*len(nDevs)))
# asdfas
	print (deviation)

fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	# if np.sum(np.isnan(t1s[i])) <1:
	# print t1s[i]
	plt.plot(t1s_times[i],t1s[i])
# print len(t1s),n,len(np.unique(allFnames))

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_times_maxtemp_plot.svg')
plt.close()

############################# plot N vs N+1 

fig,ax = plt.subplots()
for i in range(0,len(t1s)):
	for j in range(0,np.min([len(t1s[i])-1,2])):
		if (t1s_times[i][j+1]-t1s_times[i][j])>90 and (t1s_times[i][j+1]-t1s_times[i][j])<1800:
			ax.scatter(t1s[i][j],t1s[i][j+1],s=5,color='k')
ax.set_xlabel('N')
ax.set_ylabel('N+1')
#plot boundaries between each
cut1,cut2 = 27,31
ax.plot([np.min(ti1), np.max(ti1)],[cut1,cut1],color='g')
ax.plot([np.min(ti1), np.max(ti1)],[cut2,cut2],color='r')
ax.plot([cut1,cut1],[np.min(ti1), np.max(ti1)],color='g')
ax.plot([cut2,cut2],[np.min(ti1), np.max(ti1)],color='r')

#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_NN1.svg')
plt.close()


# ##let's calculate average time between deeps. 
# distAll= []
# deepsPerVideo = []
# thresh1 = 31.

# for i in range(0,len(t1s_times)):
# 	deepInds = [j for j in range(0,len(t1s[i])) if t1s[i][j]>thresh1]
# 	deepsPerVideo+=[len(deepInds)]
# 	if len(deepInds)>1:
# 		times = [t1s_times[i][j] for j in deepInds]
# 		dists = [times[j+1]-times[j] for j in range(0,len(times)-1)]
# 		distAll+= dists

# fig,ax = plt.subplots()
# sns.boxplot(np.array(['Time Between Deeps']*len(distAll)),np.array(distAll),width=0.25)
# sns.swarmplot(np.array(['Time Between Deeps']*len(distAll)),np.array(distAll),color='k')
# ax.set_ylim([0,3000])
# fig.tight_layout()
# plt.savefig(inputDir.split('/')[-2] +'_time_between_deeps.svg')


## get all first turn temps
firstTs= []
cFname = 'foo'
for i1 in range(0,len(nTs2)):
	if nFnames[i1]!= cFname and not (indsTurns[i1] in set(indsStraights)):
		firstTs.append(np.max(nDs2[i1]))
		cFname = nFnames[i1]
	elif nFnames[i1]!= cFname:
		firstTs.append(5)
		cFname = nFnames[i1]


# nTs11 = [griddata(x2*scaling,ti0[1,:],firstTs[j1]) for j1 in range(0,len(firstTs))]
nTs11 = [np.max(nTs[i]) for i in range(0,len(nTs)) if 1-np.isnan(np.max(nTs[i]))]

# #plot mean/confidence interval. 
# # print t1s
import seaborn as sns
import pandas as pd 

try:
	fig,ax = plt.subplots()

	df = pd.DataFrame(data={'Turn_Number':len(taver[0])*['T1']+['T2']*len(taver[1])+ ['T3']*len(taver[2]) +['T4']*len(taver[3]),'Temperature':taver[0]+taver[1]+taver[2]+taver[3]})
	ax = sns.boxplot(x='Turn_Number',y='Temperature',data=df)
	ax = sns.swarmplot(x='Turn_Number',y='Temperature',data=df,color='black')
	ax.set_ylim([24,40.])
	#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_first_two.svg')
	plt.close('all')


	#make the trend plot 
	xs= len(taver[0])*[0]
	xs2 = [1]*len(taver[1])
	xs3 = [2]*len(taver[2])
	xs4 = [3]*len(taver[3])
	xs5 = [4]*len(taver[4])
	xs6 = [5]*len(taver[5])
	ys = taver[0]
	ys2 = taver[1]
	ys3 = taver[2]
	ys4 = taver[3]
	ys5 = taver[4]
	ys6 = taver[5]
	print (len(xs),len(xs2),len(ys),len(ys2))
	fig,ax = plt.subplots()
	plt.scatter(xs+xs2+xs3+xs4+xs5+xs6,ys+ys2+ys3+ys4+ys5+ys6,edgecolors='black',facecolors='none')
	for i in range(0,len(ys)):
		plt.plot([xs[i], xs2[i],xs3[i],xs4[i], xs5[i],xs6[i]], [ys[i],ys2[i],ys3[i],ys4[i], ys5[i],ys6[i]],color='grey',alpha=0.5)
	ax.set_ylim([25,38.])
	#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_first_two_trend.svg')

	## combo trend/box plot
	fig,ax = plt.subplots()
	ax = sns.boxplot(x='Turn_Number',y='Temperature',data=df)
	plt.scatter(xs+xs2+xs3+xs4,ys+ys2+ys3+ys4,edgecolors='black',facecolors='none')
	for i in range(0,len(ys)):
		plt.plot([xs[i], xs2[i],xs3[i],xs4[i]], [ys[i],ys2[i],ys3[i],ys4[i]],color='grey',alpha=0.5)
	ax.set_ylim([24,40.])
	#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_first_two_trend_combo.svg')
	plt.clf()
	taverM = np.array([np.mean(taver[i1]) for i1 in range(0,len(taver))])
	taverSTD = np.array([np.std(taver[i1]) for i1 in range(0,len(taver))])
	fig,ax = plt.subplots()
	ax.plot(range(0,len(taverM)),taverM,color='blue')
	ax.fill_between(range(0,len(taverM)),taverM-taverSTD,taverM+taverSTD,color='blue',alpha=0.1)
	#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_mean_turn_temps_with_errorbars.svg')
	import os


	#csv for marco. 
	import csv
	with open('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] +'_boundary_turnsT_RAnt.csv', mode='w') as sensory_file:
		writer = csv.writer(sensory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)   
		for i in range(0,len(nTs)):
			if np.max(nTs_RA[i])>25.3 or np.max(nTs_LA[i])>25.3:
				writer.writerow(nTs_RA[i])

	with open('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] +'_boundary_turnsT_LAnt.csv', mode='w') as sensory_file2:
		writer2 = csv.writer(sensory_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for i in range(len(nTs)):
			if np.max(nTs_LA[i])>25.3 or np.max(nTs_RA[i])>25.3:
				writer2.writerow(nTs_LA[i])
except:
	pass

if 0:
	for i in range(0,len(t1s_times)):
		print (t1s_fnames[i], t1s_times[i])
		if 1 and len(t1s_times[i])>1:
			print ("python plot_traj.py " +inputDir +t1s_fnames[i] +' ' + str(t1s_start_times[i][0])+' '+ str(t1s_end_times[i][1]) +' ' + inputDir.split("/")[-2])
			os.system("python plot_traj.py " +inputDir +t1s_fnames[i] +' ' + str(t1s_start_times[i][0])+' '+ str(t1s_end_times[i][1]) +' ' + inputDir)
		#plot_boundary_behaviors(t1s_fnames[i],t1s_start_times[i],t1s_end_times[i],inputDir)

# #now do the same for the straights data
# nVs = [vList[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nrVs = [rvList[sortIndsStraights_Filt[j1]]for j1 in range(0,len(sortIndsStraights_Filt))]
# nDs = [(fSeqs[sortIndsStraights_Filt[j1]][:,1] -allCenters[sortIndsStraights_Filt[j1]][1])/scale_list[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nDs_RA = [(fSeqsRA[sortIndsStraights_Filt[j1]][:,1] -allCenters[sortIndsStraights_Filt[j1]][1])/scale_list[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nDs_LA = [(fSeqsLA[sortIndsStraights_Filt[j1]][:,1] -allCenters[sortIndsStraights_Filt[j1]][1])/scale_list[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nDs_centroid = [(fSeqsCentroid[sortIndsStraights_Filt[j1]][:,1] -allCenters[sortIndsStraights_Filt[j1]][1])/scale_list[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nbeforeListS = [nbeforeList[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]
# nafterListS = [nafterList[sortIndsStraights_Filt[j1]] for j1 in range(0,len(sortIndsStraights_Filt))]

# if inputDir.split('/')[-2][0:9] == 'output_40':
# 	nTs = [griddata(x2,ti0[1,:],nDs[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_RA = [griddata(x2,ti0[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_LA = [griddata(x2,ti0[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_centroid = [griddata(x2,ti0[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]

# elif inputDir.split('/')[-2][0:9] == 'output_30':
# 	nTs = [griddata(x2,ti[1,:],nDs[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_RA = [griddata(x2,ti[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_LA = [griddata(x2,ti[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_centroid = [griddata(x2,ti[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]

# elif inputDir.split('/')[-2][0:9] == 'output_35':
# 	nTs = [griddata(x2,ti35[1,:],nDs[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_RA = [griddata(x2,ti35[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_LA = [griddata(x2,ti35[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_centroid = [griddata(x2,ti35[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]

# elif inputDir.split('/')[-2][0:9] == 'output_25':
# 	nTs = [griddata(x2,ti0[1,:],nDs[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_RA = [griddata(x2,ti0[1,:],nDs_RA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_LA = [griddata(x2,ti0[1,:],nDs_LA[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]
# 	nTs_centroid = [griddata(x2,ti0[1,:],nDs_centroid[j1]) for j1 in range(0,len(sortIndsStraights_Filt))]

# else:
# 	nTs = False

# if nTs:
# 	fout = inputDir.split('/')[-2] + '_boundary_straightsData.pkl'
# 	w1 = open(fout,"wb")
# 	datOut = (nVs,nrVs,nDs,nTs,nTs_RA,nTs_LA,nTs_centroid,nbeforeListS,nafterListS)
# 	pickle.dump(datOut,w1)
# 	w1.close()

# with open(inputDir.split('/')[-2] +'_boundary_straightsT_RAnt.csv', mode='w') as sensory_file:
# 	writer = csv.writer(sensory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 	for i in range(0,len(nTs)):
# 		writer.writerow(nTs_RA[i])

# with open(inputDir.split('/')[-2] +'_boundary_straightsT_LAnt.csv', mode='w') as sensory_file:
# 	writer = csv.writer(sensory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 	for i in range(0,len(nTs)):
# 		writer.writerow(nTs_LA[i])

############### make fancy histogram

numCrosses = len(sortIndsStraights_Filt)

tmax = griddata(x2,ti1[1,:],np.min(yvals)+5) 


#remove some of the levels 

#filter 1
# filt1 = [0,3,6,14]
if 1:
	yvals1 = yvals
	levels1 = levels
	filt2 = [0,2,6,14]
	if yvals == yvals30:
		filt2 = filt2[0:3]
	yvals = [yvals[f2] for f2 in filt2]
	levels = [levels[f2] for f2 in filt2]

# print yvals40
# print levels
# print maxTsTurns
fig,ax = plt.subplots()
for j1 in range(0,len(yvals)):
	if j1%6 ==0:
		ax.plot([0,40],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
	else:
		ax.plot([0,40],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
n1,bins1,patches1 = ax.hist(maxTsTurns,bins=list(np.array(yvals))+[np.min(yvals)+5]+[4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20)

ax.hist([5]*numCrosses,bins=[4.5,5.5],color=col1,edgecolor='black',linewidth=1.5,zorder=2,orientation='horizontal')
intInds = np.digitize(maxTsTurns+[5]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5]+[4.5,5.5])

#to plot first turn temps. 
ax.hist(firstTs,bins=list(np.array(yvals))+[np.min(yvals)+5]+[4.5,5.5],color='green',edgecolor='black',linewidth=1.5,zorder=100,orientation='horizontal')
ax.set_yticks(list(np.array(yvals))+[np.min(yvals)+5,5])
ax.set_yticklabels(list(levels)+[np.around(tmax,decimals=3),'Crosses'])
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_hists_maxT.svg')

pickle.dump([n1,numCrosses,bins1,maxTsTurns],open('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2] +'_hist_data.pkl','wb'))


plt.close()



fig,ax = plt.subplots(figsize=(5,10))
# for j1 in xrange(0,len(yvals)):
# 	if j1%6 ==0:
# 		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
# 	else:
# 		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
ax.hist(maxTsTurns+[5]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5,4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[5]*numCrosses)/float(len(maxTsTurns+[5]*numCrosses)))
# ax.hist([25]*numCrosses,bins=[24,26],color=col1,edgecolor='black',linewidth=1.5,zorder=0,orientation='horizontal')
ax.set_yticks(list(np.array(yvals))+[np.min(yvals)+5,5])
ax.set_yticklabels(list(levels)+[np.around(tmax,decimals=3),'Crosses'])
# ax.set_xlim([0,0.65])
ax.set_xlim([0,0.45])
ax.set_ylim([-3,6])
fig.tight_layout()
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_maxT.svg')
plt.close()
######################### make fraction of first turns divided by all turns bar plot. 


# fig,ax = plt.subplots()
# for j1 in xrange(0,len(yvals)):
# 	if j1%6 ==0:
# 		ax.plot([0,1.],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
# 	else:
# 		ax.plot([0,1.],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
#get bin we're in. 
ftInds = np.digitize(firstTs,bins=list(np.array(yvals))+[np.min(yvals)+5])-1
nfts = [n1[ftInds[i]] for i in range(0,len(ftInds))]
#scale by turns in the bin.
weights = np.zeros_like(firstTs)
ones1 = np.ones_like(firstTs) 
for i in range(0,len(ftInds)):
	if nfts[i]!=0:
		weights[i]=ones1[i]/nfts[i]
	else:
		weights[i]=0.

# ax.hist(firstTs,bins=list(np.array(yvals))+[np.min(yvals)+5]+[4.5,5.5],color='green',edgecolor='black',linewidth=1.5,zorder=100,orientation='horizontal',weights=weights)
# # ax.hist(maxTsTurns+[25]*numCrosses,bins=list(np.array(yvals))+[20,24,26],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[25]*numCrosses)/float(len(maxTsTurns+[25]*numCrosses)))
# # ax.hist([25]*numCrosses,bins=[24,26],color=col1,edgecolor='black',linewidth=1.5,zorder=0,orientation='horizontal')
# ax.set_yticks(list(np.array(yvals))+[5])
# ax.set_yticklabels(list(levels))
# plt.savefig(inputDir.split('/')[-2] +'_fracs_first_maxT.svg')
# plt.close()


#make fracs plot with max turns 
fig,ax = plt.subplots(figsize=(5,6.75))
'''
for j1 in range(0,len(yvals)):
	if j1%6 ==0:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
	else:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
'''
cts,bins,p0 = ax.hist(maxTsTurns+[yvals[0]+5.2]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5,yvals[0]+5.4],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[5]*numCrosses)/float(len(maxTsTurns+[5]*numCrosses)))
ax.set_yticks(list(np.array(yvals1))+[np.min(yvals)+5,yvals[0]+5.2])
ax.set_yticklabels(list(levels1)+[np.around(tmax,decimals=3),'Crosses'])
ax.set_xlim([0,0.6])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim([yvals[0]-0.1,yvals[0]+5.8])
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_withfirst_maxT.svg')
plt.close() 

# two bar plot
fig,ax = plt.subplots(figsize=(10,5))
cts,bins,p0 = ax.hist(maxTsTurns+[5]*numCrosses, bins=[yvals[0],yvals[2]]+[4,4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[5]*numCrosses)/float(len(maxTsTurns+[5]*numCrosses)))
ax.set_yticks([yvals[0],yvals[2]]+[4,5])
ax.set_yticklabels([levels[0], levels[2],'hot','crosses'])
ax.set_xlim([0,1])
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_marcoplot.svg')
plt.close()

# Plotting others
maxTsTurnsTemp = griddata(x2,ti1[1,:],np.array(maxTsTurns))
fig,ax = plt.subplots(figsize=(10,5))
ax = sns.violinplot(y=maxTsTurnsTemp,bw='scott',orient='v',cut=0, color=col1)
ax.set_yticks(list(levels)+[tmax])
ax.yaxis.grid(True, which='major')
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_marcoplot2.svg')
plt.close()

# plotting violin over turn bar
fig,ax = plt.subplots(figsize=(5,10))
sns.violinplot(y = maxTsTurns, bw='scott', orient='v', cut=0, color=col1, ax = ax)
plt.setp(ax.collections, alpha=0.7)
cts,bins,p0 = ax.hist(maxTsTurns+[5]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5,4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=0,weights=np.ones_like(maxTsTurns+[5]*numCrosses)/float(len(maxTsTurns+[5]*numCrosses)))
ax.set_yticks(list(np.array(yvals))+[np.min(yvals)+5,5])
ax.set_yticklabels(list(levels)+[np.around(tmax,decimals=3),'Crosses'])
ax.set_xlim([0,0.6])
ax.set_xticks(np.arange(7)*0.1)
ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim([-3,6])
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_viol_eturns.svg')
plt.close() 
print('cts=',cts)




'''
# make fracs plot with first turns above and below 0 (crosses included)
firstTsFake = firstTs.copy()
for kk in range(len(firstTs)):
	if firstTs[kk]==5:
		firstTsFake[kk]=1
kk = np.ceil(np.max(np.abs(maxTsTurns)))

fig,ax = plt.subplots(figsize=(10,5))

cts,bins,p0 = ax.hist(maxTsTurns+[1]*numCrosses,bins=[-kk,0,kk],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[1]*numCrosses)/float(len(maxTsTurns+[1]*numCrosses)))
ax.hist(firstTsFake,bins=bins,orientation='horizontal',color='green',edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(firstTsFake)/float(len(maxTsTurns+[1]*numCrosses)))
ax.set_yticks([-kk,0,kk])
ax.set_yticklabels([25,'border','HOT'])
ax.set_xlim([0,1])
ax.set_ylim([-kk-1,kk+1])
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_just_border.svg')
plt.close()

# make fracs plot with first turns above and below 0 (crosses excluded)
firstTsFake = firstTs.copy()
try:
    while True:
        firstTsFake.remove(5)
except ValueError:
    pass

fig,ax = plt.subplots(figsize=(10,5))

cts,bins,p0 = ax.hist(maxTsTurns,bins=[-kk,0,kk],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns)/float(len(maxTsTurns)))
ax.hist(firstTsFake,bins=bins,orientation='horizontal',color='green',edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(firstTsFake)/float(len(maxTsTurns)))
ax.set_yticks([-kk,0,kk])
ax.set_yticklabels([25,'border','HOT'])
ax.set_xlim([0,1])
ax.set_ylim([-kk-1,kk+1])
fig.tight_layout()
plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_just_border_crosses_excluded.svg')
plt.close()

'''

#make fracs plot with first turns 
fig,ax = plt.subplots(figsize=(5,10))
for j1 in range(0,len(yvals)):
	if j1%6 ==0:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
	else:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
ax.hist(maxTsTurns_1+[5]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5,4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns_1+[5]*numCrosses)/float(len(maxTsTurns_1+[5]*numCrosses)))
ax.set_yticks(list(np.array(yvals))+[np.min(yvals)+5,5])
ax.set_yticklabels(list(levels)+[np.around(tmax,decimals=3),'Crosses'])
ax.set_xlim([0,0.65])
ax.set_ylim([-3,6])
fig.tight_layout()
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_fullMax.svg')
plt.close()

#sideways bar plots, where max antennal temp difference is shown 
fig,ax = plt.subplots(figsize=(5,10))
dInds = np.digitize(maxTsTurns,bins=list(np.array(yvals))+[np.min(yvals)+5,4.5,5.5])
boxDiffs= [[] for i in range(np.max(dInds))]
for j1 in range(0,len(maxTsTurns)):
	boxDiffs[dInds[j1]-1].append(maxTsTurns_Diff[j1])

ax.boxplot(boxDiffs)
ax.set_ylim([0,2.5])
ax.set_ylabel('Temperature difference between antennae (C)')
ax.set_xlabel('Box number')
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_boxdiffs.svg')

######## plot distribution of first turns
#make fracs plot with first turns 
fig,ax = plt.subplots(figsize=(5,10))
for j1 in range(0,len(yvals)):
	if j1%6 ==0:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='red',linewidth=1.5,alpha=1.,zorder=10)
	else:
		ax.plot([0,0.5],[yvals[j1],yvals[j1]],color='orange',linewidth=1.5,alpha=1.,zorder=10)
# ax.hist(maxTsTurns+[5]*numCrosses,bins=list(np.array(yvals))+[np.min(yvals)+5,4.5,5.5],orientation='horizontal',color=col1,edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(maxTsTurns+[5]*numCrosses)/float(len(maxTsTurns+[5]*numCrosses)))
ax.hist(firstTs,bins=list(np.array(yvals))+[np.min(yvals)+5]+[4.5,5.5],orientation='horizontal',color='green',edgecolor='black',linewidth=1.5,zorder=20,weights=np.ones_like(firstTs)/float(len(firstTs)))

ax.set_yticks(list(np.array(yvals))+[np.min(yvals)+5,5])
ax.set_yticklabels(list(levels)+[np.around(tmax,decimals=3),'Crosses'])
ax.set_xlim([0,0.65])
ax.set_ylim([-3,6])
fig.tight_layout()
#plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+inputDir.split('/')[-2] +'_fracs_just_first.svg')

########################
# now, plot zoomed in versions of trajectories
plt.close('all')
indsTurns = indsTurnsAndStraights
nDs2 = [(fSeqs[indsTurns[j1]][:,1] -allCenters[indsTurns[j1]][1])/scale_list[indsTurns[j1]] for j1 in range(0,len(indsTurns)) ]
nFnames = [allFnames[ indsTurns[j1]] for j1 in range(0,len(indsTurns)) ]
startListT2 = [start_time_list[indsTurns[j1]] for j1 in range(0,len(indsTurns))]

if inputDir.split('/')[-2][0:9] == 'output_40':
	nTs2 = [griddata(x2,ti0[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]

elif inputDir.split('/')[-2][0:9] == 'output_30':
	nTs2 = [griddata(x2,ti[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]

elif inputDir.split('/')[-2][0:9] == 'output_35':
	nTs2 = [griddata(x2,ti35[1,:],nDs2[j1]) for j1 in range(0,len(indsTurns))]

if 0:
	import matplotlib as mpl
	C = np.loadtxt('/home/josh/Desktop/heat_transfer/3D_static/cmap.txt',dtype=np.int).astype('float')
	cm1 = C/255.0
	cm1 = mpl.colors.ListedColormap(cm1)

	allMaxes2 = []
	allMaxes_Ind = []
	allHeights = []
	v_data = []
	loc_list,loc_times = [],[]
	tVals_before = []
	tVals_beforeRA,tVals_beforeLA = [],[]
	if len(indsTurns)>0:
		numRows = len(indsTurns)//20+1 if len(indsTurns) %20 !=0 else len(indsTurns)//20
		# spacings =[]
		# for i1 in xrange(0,len(widths)):
		# 	spacings.append((bigWidth-(2.+widths[i1]))/20.)
		newX = 4
		added=0
		import peakutils
		#### first make staff figure for turns. 
		for i1 in range(0,len(indsTurns)):
			sI = indsTurns[i1]
			if np.sum(np.abs(rvList[sI]))==0.:
				continue
			cSequence = fSeqs[sI]
			cSequenceRA = fSeqsRA[sI]
			cSequenceLA = fSeqsLA[sI]
			cFname = nFnames[i1]
			# locI = np.argmax(cSequence[:,1] - allCenters[sI][1])
			locI = np.argmin(cSequence[:,0])
			max1 = np.max(cSequence[:,1] - allCenters[sI][1])
			max1_arg = np.argmax(cSequence[:,1] - allCenters[sI][1])
			allMaxes2.append(max1/scale_list[sI])
			allMaxes_Ind+=[cSequence[max1_arg-3,1]- allCenters[sI][1],cSequence[max1_arg-2,1]- allCenters[sI][1],cSequence[max1_arg-1,1]- allCenters[sI][1],cSequence[max1_arg,1]- allCenters[sI][1]]
			allHeights.append(cSequence[:,1] - allCenters[sI][1])
			try:
				pIs = peakutils.indexes(np.abs(rvList[sI]),thres=0.3,min_dist=4)
				pI = np.argmin(np.abs(pIs -max1_arg))
				argRVmax = pIs[pI]
			except Exception as ex:
				# print ex
				# plt.plot(rvList[sI])
				# plt.show()
				argRVmax = np.argmax(np.abs(rvList[sI]))
			v_data.append([vList[sI][argRVmax],rvList[sI][argRVmax]])
			locX = cSequence[locI,0]/scale_list[sI]
			widthX = (np.max(cSequence[:,0]) - np.min(cSequence[:,0]))/scale_list[sI]
			#plot trajectory
			cSequence[:,0] = cSequence[:,0]/scale_list[sI] - locX + newX
			loc_list.append(newX)
			loc_times.append(np.argmax(nTs2[i1])+startListT2[i1])
			#flip x coord to match video.
			minXc = np.min(cSequence[:,0])
			maxXc = np.max(cSequence[:,0])

			cSequence[:,0] = maxXc - cSequence[:,0] + minXc 

			tVals_before.append(griddata(x2,ti1[1,:],(cSequence[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))
			tVals_beforeLA.append(griddata(x2,ti1[1,:],(cSequenceLA[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))
			tVals_beforeRA.append(griddata(x2,ti1[1,:],(cSequenceRA[0:max1_arg,1]-allCenters[sI][1])/scale_list[sI]))

			newX+= widthX+3.
			line, = ax.plot(cSequence[:,0],(cSequence[:,1]-allCenters[sI][1])/scale_list[sI],color='black',linewidth=0.2)
			ax.scatter(np.array(cSequence[:,0]),np.array((cSequence[:,1]-allCenters[sI][1]))/scale_list[sI],c=np.array(np.abs(vList[sI]))/scale_list[sI],cmap=plt.cm.coolwarm,s=2,vmin=0,vmax=1./3.)
			add_arrow_to_line2D(ax,line,arrow_locs=[0,0.5,1])
			try:
				if i1==len(indsTurns)-1:
					finish=1
				elif nFnames[i1+1]!=cFname:
					finish =1
				else:
					finish=0
			except:
				finish=1
			#if new video, print out the old ones, and 
			if finish:
				if inputDir.split('/')[-2][0:9] == 'output_40':
					for j1 in range(0,len(levels40)):
						ax.plot([0,newX],[yvals40[j1],yvals40[j1]],color='black',linewidth=0.1)
					ax.imshow(ti0.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2),np.max(x2)])
					tVals = griddata(x2,ti0[1,:],allMaxes2)
					tlabels = np.arange(25.5,37.5,1.0)
					sVals = griddata(ti0[1,:],x2,tlabels)
					dtVals = griddata(x2,np.gradient(ti0[1,:]),allMaxes2)
					tVals_dt = griddata(x2,ti0[1,:],allMaxes_Ind)


				if inputDir.split('/')[-2][0:9] == 'output_30':
					for j1 in range(0,len(levels30)):
						ax.plot([0,newX],[yvals30[j1],yvals30[j1]],color='black',linewidth=0.5)
					ax.imshow(ti.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2),np.max(x2)])

					tVals = griddata(x2,ti[1,:],allMaxes2)
					tVals_dt = griddata(x2,ti[1,:],allMaxes_Ind)
					tlabels = np.arange(25.5,29.5,1.0)
					sVals = griddata(ti[1,:],x2,tlabels)
					dtVals = griddata(x2,np.gradient(ti[1,:]),allMaxes2)


				if inputDir.split('/')[-2][0:9] == 'output_35':
					for j1 in range(0,len(levels35)):
						ax.plot([0,newX],[yvals35[j1],yvals35[j1]],color='black',linewidth=0.5)
					ax.imshow(ti35.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2),np.max(x2)])
						
					tVals = griddata(x2,ti35[1,:],allMaxes2)
					tlabels = np.arange(25.5,33.5,1.0)
					sVals = griddata(ti35[1,:],x2,tlabels)
					dtVals = griddata(x2,np.gradient(ti35[1,:]),allMaxes2)
					tVals_dt = griddata(x2,ti35[1,:],allMaxes_Ind)
				if inputDir.split('/')[-2][0:9] == 'output_25':
					for j1 in range(0,len(levels35)):
						ax.plot([0,newX],[yvals35[j1],yvals35[j1]],color='black',linewidth=0.5)
						ax.imshow(ti35.T,vmin=25.,vmax=40.,cmap=cm1,origin='lower',extent=[0,newX,np.min(x2),np.max(x2)])
						
						tVals = griddata(x2,ti35[1,:],allMaxes2)
						tlabels = np.arange(25.5,33.5,1.0)
						sVals = griddata(ti35[1,:],x2,tlabels)
						dtVals = griddata(x2,np.gradient(ti35[1,:]),allMaxes2)
						tVals_dt = griddata(x2,ti35[1,:],allMaxes_Ind)

				ax.set_xticks(loc_list)
				ax.set_xticklabels(loc_times)
				ax.set_aspect('equal','box')
				ax.set_anchor('W')
				ax.set_ylim([-10,10])
				fig.tight_layout()
				plt.savefig('bdryflex/bdryflex_plots/bdryflex_plots_'+groupName+'/'+'traj_plots/'+inputDir + '_'+cFname+'_turn_plot.svg')
				plt.close()
				fig,ax = plt.subplots()
				if i1!=len(indsTurns)-1:
					cFname=nFnames[i1+1]
					loc_list,loc_times=[],[]
					newX=4

# For Jenna, maxTsTurns now has a flyId associated with it (thank god)

belowTurn = [1*(maxTsTurns[i]< yvals[2]) for i in range(len(maxTsTurns))]
dp = pd.DataFrame({'below': belowTurn, 'maxT': maxTsTurns, 'flyid': maxTsNames} )
dp.to_csv('bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/'+inputDir.split('/')[-2]+'jenna_maxturns.csv')