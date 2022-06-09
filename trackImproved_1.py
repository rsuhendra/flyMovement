#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: josh

"""

import copy
from matplotlib.pyplot import show
import peakutils


def findCOM(binImage, vDims):
    # function to calculate the center of mass of the fly.
    # calculates marginal probabilities and then finds the highest probability location.
    binImage = np.round(binImage / float(255))
    p1_x = np.sum(binImage, 1)  # marginals
    tot = np.sum(p1_x)
    p1_x = p1_x/tot
    p1_y = np.sum(binImage, 0)/tot
    # means
    # print p1_x
    cx = np.sum(p1_x*np.arange(vDims[1]))
    cy = np.sum(p1_y*np.arange(vDims[0]))
    return (cx, cy)


def movingaverage(interval, window_size):
<<<<<<< Updated upstream
	#function to calculate a moving average of a trajectory. 
	#Note: at the moment, this should only be used with odd window sizes. 
	if np.size(interval)>0:
		window= np.ones(int(window_size))
		averaged = np.convolve(interval, window, 'same')/float(window_size)
		for j in xrange(0,(window_size-1)/2):
			# print j
			averaged[j] = np.sum(interval[0:2*j+1])/(2.*j + 1.)
		for j in xrange(-(window_size-1)/2-1,0):
			# print j,averaged[j:]
			averaged[j] = np.sum(interval[2*(j+1)-1:])/(-2.*(j+1)+1)
	else:
		averaged = []
	return averaged

def coreDataExtract(filename):
	#this function performs the data extraction from the video, converting the fly to a binary image
	# and then performing a least squares ellipse fit to find the orientation of the fly. 
	# A few different centroid calculations can be done. At the moment, the centroid is taken as the 
	# center of mass of the fly. This will be updated soon however. 
	import imageio
	import cv2
	vid = imageio.get_reader(filename,  'ffmpeg')
	fps = vid.get_meta_data()['fps']
	vDims =vid.get_meta_data()['source_size']
	vLength = vid.get_length()
	nStart =0
	nFinish = vLength
	nums = xrange(nStart,nFinish)

	allThetas = np.array([])
	allCenters = np.array([])
	allCOM = np.array([])
	al1s = np.array([])
	al2s = np.array([])
	listThetas = []
	listCenters = []
	listCOM = []
	listAL1 = []
	listAL2 = []
	# cDir = theta
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	frames = []
	badCount = 0
	#print vDims
	#print kernel
	mask = np.ones(vDims,dtype="uint8")*255
	for i in xrange(0,vDims[0]):
		for j in xrange(0,vDims[1]):
			if np.sqrt((q4[0]-i)**2 + (q4[1]-j)**2) > ((halfWidth + halfHeight)/2):
				mask[i,j] = 0

	a1,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
	#mask.tolist()
		#just need to make sure that the first frame is a good shot... otherwise there can be a problem.
	for num in nums:
		im5 = vid.get_data(num)
		# if num ==0:
		# 	plt.imshow(im5)
		imageG5_1 = cv2.cvtColor(im5,cv2.COLOR_BGR2GRAY)
		# imageG5_1 = cv2.bitwise_and(imageG5,imageG5,mask=mask)#[25:200,25:200]
		r1,t5_1 = cv2.threshold(imageG5_1,200, 255, cv2.THRESH_BINARY)
		# tOld = t5_1
		t5_1 = cv2.bitwise_and(mask.T,t5_1,t5_1)#[25:200,25:200]
		# asdfas
	   #t5_1 = cv2.dilate(t5_1,kernel,iterations=1)

		t5_1 = cv2.erode(t5_1,kernel,iterations = 1)
		t5_1 = cv2.dilate(t5_1,kernel,iterations = 1)
		tOld = t5_1
		t5_1 = cv2.bitwise_not(t5_1)
		# t5_1 = cv2.erode(t5_1,kernel,iterations = 1)
		# t5_1 = cv2.dilate(t5_1,kernel,iterations = 1)
		# cv2.imshow("BLA",t5_1)
		# cv2.waitKey(50)
		im2, contours,hierarchy = cv2.findContours(t5_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# if len(frames)>1 and num>600:
		# cv2.imshow("BLA",t5_1)
		# cv2.waitKey(100)
		# 	print frames[len(frames)-1],frames[len(frames)-2],num

		if len(contours) < 2:
			print num, "problem frame:insufficient data"
			if len(allThetas)>0:
				listThetas.append(allThetas)
				listCenters.append(allCenters)
				listCOM.append(allCOM)
				listAL1.append(al1s)
				listAL2.append(al2s)
				#print len(allThetas), len(al1s)
				allThetas = np.array([])
				allCenters = np.array([])
				allCOM = np.array([])
				al1s = np.array([])
				al2s = np.array([])
				badCount=0
				frames.append(num)
		else:
			cnt = contours[1]
			if len(cnt)<5:
				# print len(cnt),num
				if np.size(allThetas)>0:
					if len(frames)%2 ==0:
						frames.append(num)
					allThetas = np.append(allThetas,theta)
					allCenters = np.append(allCenters,[cx,cy])
					allCOM = np.append(allCOM,cCOM)
					al1s = np.append(al1s,al1)
					al2s = np.append(al2s,al2)
			else:
				ellipse = cv2.fitEllipse(cnt)
				cx,cy = ellipse[0]
				al1,al2 = ellipse[1]
				theta = ellipse[2]
				if centerType =='COM':
					cCOM = findCOM(tOld,vDims)
				else:
					cCOM = [cy,cx]

				# cCOM1 = [cCOM[1],cCOM[0]]
				# t5_1 = cv2.circle(t5_1,tuple([int(i1) for i1 in cCOM1]),1,0,-1)
				# cv2.imshow("masked",t5_1)
				# cv2.resizeWindow('masked', 600,600)
				# cv2.waitKey(50)
				if (al1/al2 > 0.6): # (np.sqrt(((cCOM[0]-q4[0])/(halfWidth-10))**2 + ((cCOM[1]-q4[1])/(halfHeight-10)**2)) > 1):
					if badCount>10:
						listThetas.append(allThetas)
						listCenters.append(allCenters)
						listCOM.append(allCOM)
						listAL1.append(al1s)
						listAL2.append(al2s)
						allThetas = np.array([])
						allCenters = np.array([])
						allCOM = np.array([])
						al1s = np.array([])
						al2s = np.array([])
						frames.append(num)
						badCount = 0
					elif np.size(allThetas) == 0:
						badCount = 0
						# print "still in bad region",num
						# if len(frames)%2 ==1:
						#     frames = []
					else:   
						if len(frames)%2 ==0:
							frames.append(num)
						if num ==nStart:
							theta_old = theta

						if centerType =='COM':
							allCOM = np.append(allCOM,cCOM)
						else:
							allCOM = np.append(allCOM,[cy,cx])
						# allThetas = np.append(allThetas,theta)
						allCenters = np.append(allCenters,[cx,cy])
						al1s = np.append(al1s,al1)
						al2s = np.append(al2s,al2)
						allThetas = np.append(allThetas,theta_old)
						badCount +=1

				else:
					# if len(frames)%2 ==0:
					#     frames.append(num)
					if np.size(allThetas) == 0:
						frames.append(num)
					if centerType =='COM':
						allCOM = np.append(allCOM,cCOM)
					else:
						allCOM = np.append(allCOM,[cy,cx])
					# allThetas = np.append(allThetas,theta)
					allCenters = np.append(allCenters,[cx,cy])
					al1s = np.append(al1s,al1)
					al2s = np.append(al2s,al2)
					allThetas = np.append(allThetas,theta)
					theta_old = theta
			# if num == 2091:
			#     asdfas

	listThetas.append(allThetas)
	listCenters.append(allCenters)
	listCOM.append(allCOM)
	listAL1.append(al1s)
	listAL2.append(al2s)
	if len(frames)%2 ==1:
		frames.append(num+1)
	COM_x1 = []
	COM_y1 = []
	vx_1 = []
	vy_1 = []
	denoisedThetas =[]
	denoisedThetas_deriv = []
	# print len(listThetas),frames,len(frames)
	# asdf
	for j in xrange(0,min(len(listThetas),len(frames)/2)):
		if len(listThetas[j]) < 11:
			k1 = int(np.floor(len(listThetas[j])/2) +1)
			k2 = k1
		else:
			k1 = 11
			k2 = 9

		# if np.sum(np.array(listCOM[j][0::2])==0)>0:
		# 	print listCOM[j][1::2]
		# 	asdfa

		COM_y1.append(movingaverage(listCOM[j][1::2],k1))
		COM_x1.append(movingaverage(listCOM[j][0::2],k1))
		if k1 ==1:
			vx_1.append(np.zeros(1))
			vy_1.append(np.zeros(1))
		else:
			vx_1.append(movingaverage(np.gradient(COM_x1[j]),k2)) 
			vy_1.append(movingaverage(np.gradient(COM_y1[j]),k2))
		lStart = frames[2*j]
		lFinish =  frames[2*j+1]
		numsMinusZero = xrange(1,lFinish-lStart)
		allDiffs = np.zeros(lFinish-lStart)
		# print j, listThetas[j],lStart,lFinish
		allDiffs[0] = listThetas[j][0]
		for num in numsMinusZero:
			theta = listThetas[j][num]
			cDir = listThetas[j][num-1]
			mi2 = np.argmin([abs(theta-cDir),abs(180+theta-cDir),abs(-180+theta-cDir)])

			change = np.min([abs(theta-cDir),abs(180+theta-cDir),abs(-180+theta-cDir)])
			bla = [(theta-cDir),(180+theta-cDir),(-180+theta-cDir)]
			allDiffs[num]=allDiffs[num-1] + change*np.sign(bla[mi2])

		# denoisedThetas.append(allDiffs)
		# denoisedThetas_deriv.append(np.gradient(allDiffs,9))

		denoisedThetas.append(movingaverage(allDiffs,k2))
		if k1 ==1:
			denoisedThetas_deriv.append(np.zeros(1))
		else:
			denoisedThetas_deriv.append(movingaverage(np.gradient(movingaverage(allDiffs,k2)),k2))


	# plt.plot(al1s/al2s,color='blue')
	# #plt.plot(al2s,color = 'red')
	# plt.show()

	#now calculate some statistics about hot/cold regions
	quad = []
	qH = []
	qB = []
	for j in xrange(0,min(len(listThetas),len(frames)/2)):
		lStart = frames[2*j]
		lFinish =  frames[2*j+1]
		nums121 = xrange(0,lFinish-lStart)
		quad_1 = np.zeros(lFinish-lStart)
		quad_head = np.zeros(lFinish-lStart)
		quad_back = np.zeros(lFinish-lStart)

		for ind in nums121:
			quad_1[ind] = calcQuadrant(COM_y1[j][ind],COM_x1[j][ind])
			headPos = headPosition(COM_y1[j][ind],COM_x1[j][ind],denoisedThetas[j][ind])
			backPos = backPosition(COM_y1[j][ind],COM_x1[j][ind],denoisedThetas[j][ind])

			# plt.plot(headPos[0],headPos[1],'ro')
			# plt.plot(backPos[0],backPos[1],'bo')
			# plt.plot(COM_y1[j][ind],COM_x1[j][ind],'go')
			# plt.show()

			quad_head[ind] = calcQuadrant(headPos[0],headPos[1])
			quad_back[ind] = calcQuadrant(backPos[0],backPos[1])

		quad.append(quad_1)
		qH.append(quad_head)
		qB.append(quad_back)

	return (COM_x1,COM_y1, vx_1,vy_1,denoisedThetas,denoisedThetas_deriv,nStart,nFinish,listAL1,listAL2,frames,quad,qH,qB)


def headPosition(x_p,y_p,theta_p):
	cToHead = np.array([np.sin(np.pi/180.*theta_p),np.cos(np.pi/180.*theta_p)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_p,y_p])+7.0*cToHead
	return cVector

def backPosition(x_p,y_p,theta_p):
	cToHead = np.array([np.sin(np.pi/180.*theta_p),np.cos(np.pi/180.*theta_p)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_p,y_p])-7.0*cToHead
	return cVector


class simpleBehaviorEvent:
	#elementary behavioral event class. The event class used now. 
	def __init__(self,fStart):
		self.frameStart = fStart
		self.modes = []
		self.quality = 0
		self.onWall = 0
		self.distTraveled = 0.0
		self.quadrants = []
		self.quadrants_head = []
		self.quadrants_back = []
		self.nearTempBarrier =-180
		self.minDistTempB = 1000
		self.arenaFile = arenaFile
		self.showQuadrants = showQuadrants
		self.frameEntryT = -1
	def addModeType(self,mode):
		self.modes = np.append(self.modes,mode)

	def getModes(self):
		return self.modes
	def getStartAndFinish(self):
		return [self.frameStart,self.frameEnd]
	def markQualityIsBad(self):
		self.quality +=1
	def markOnWall(self):
		self.onWall = 1
	def markNearTempBarrier(self,ang,fNum):
		if self.nearTempBarrier == -180:
			self.nearTempBarrier = ang
			self.frameEntryT = fNum
	def minTempB_Dist(self,cDist):
		if cDist < self.minDistTempB:
			self.minDistTempB = cDist
	def addQuadrant(self,qNum):
		if qNum not in self.quadrants:
			self.quadrants.append(qNum)
	def addQuadrant_head(self,qNum):
		if qNum not in self.quadrants_head:
			self.quadrants_head.append(qNum)
	def addQuadrant_back(self,qNum):
		if qNum not in self.quadrants_back:
			self.quadrants_back.append(qNum)
	def addMaxProj(self,maxProj):
		buff = StringIO.StringIO()
		if self.frameEnd-self.frameStart>=0:
			img1 = Image.fromarray(maxProj/(self.frameEnd-self.frameStart +1))
		else:
			img1 = Image.fromarray(maxProj/(self.frameEnd-self.frameStart +2))
		img1 = img1.convert('RGB')
		img1.save(buff,"JPEG",quality=100)
		self.maxProj = buff
	def endEvent(self,fEnd):
		self.frameEnd = fEnd

def decomposeVelocity(vx_1,vy_1,denoisedThetas):
	#projects the translational velocity onto a new coordinate system
	#where rather than x and y velocity, we have forward/backward and left/right slip velocity 
	transV = []
	slipV = []
	for j in xrange(0,len(vx_1)):
		listLength = len(vx_1[j])
		nums2 = xrange(0,listLength)
		# print listLength,len(vx_1),j
		transV.append(np.zeros(listLength))
		slipV.append(np.zeros(listLength))
		# print listLength, len(denoisedThetas[j])
		for num in nums2:
			# print listLength,len(vx_1),j,num,len(denoisedThetas[j])
			trueV = np.array([vy_1[j][num],vx_1[j][num]])
			cDir = np.array([-np.sin(np.pi*denoisedThetas[j][num]/180.),np.cos(np.pi*denoisedThetas[j][num]/180.)])
			cDir_tang = np.array([np.cos(np.pi*denoisedThetas[j][num]/180.),np.sin(np.pi*denoisedThetas[j][num]/180.)])
			transV[j][num] = np.dot(trueV,cDir)
			slipV[j][num] = np.dot(trueV,cDir_tang)
		# plt.plot(transV[j])
		# plt.plot(slipV[j])
		# plt.show()
	return (transV,slipV)

def checkOrientation(transV,denoisedThetas1):
	redo = 0
	for j in range(0,len(transV)):
		if np.sum(transV[j])<0 or np.sum([tV1 <-0.01 for tV1 in transV[j]])>np.sum([tV1 >0.01 for tV1 in transV[j]]):
			denoisedThetas1[j] = [dT + 180 for dT in denoisedThetas1[j]]
			redo = 1

	return (redo,denoisedThetas1)

def timepointClassify(transV,denoisedThetas_deriv,listAL1,listAL2,COM_x1,COM_y1,frames):
	#this function converts translational/rotational information into an behavioral mode for that frame.
	# the output of this is fed into the eventTabulate function to form "behavioral" events. 
	transThresh = 0.15
	slipThresh = 0.15
	rotThresh = 1.5
	mode = np.ones(nFinish-nStart)*-1
	modeQuality = np.zeros(nFinish-nStart)
	onWall = np.ones(nFinish-nStart)*1
	nearTempB = np.ones(nFinish-nStart)*-180.
	nearTempB_dist = np.ones(nFinish-nStart)*-1.
	#q4 = [99,95]
	# vid = imageio.get_reader(filename,  'ffmpeg')#for demoing. 
	print "===List of modes occuring in every frame==="
	###now, let's classify our "events". these can be left and right turns(stationary or directed),forward and backward movements, or rests. 
	for j in xrange(0,len(transV)):
		listLength = len(transV[j])
		nums2 = xrange(0,listLength)
		n1 = frames[2*j]
		for num in nums2:
			[inPosTrans,inNegTrans,inSlip,inRotRight,inRotLeft] = [0,0,0,0,0]
			if transV[j][num] >transThresh:
				inPosTrans = 1
			if transV[j][num] <-transThresh:
				inNegTrans = 1
			if abs(slipV[j][num])>slipThresh:
				inSlip = 1
			if  denoisedThetas_deriv[j][num]> rotThresh:
				inRotRight = 1
			if  denoisedThetas_deriv[j][num]<-rotThresh:
				inRotLeft = 1
			dsum = np.sum([inPosTrans,inNegTrans,inSlip,inRotRight,inRotLeft])
			if dsum==0:
				inRest = 1
				mode[n1] = 0
				# print "rest"
			elif inPosTrans and inRotLeft:
				mode[n1] = 1
				# print "forward, left"
			elif inPosTrans and inRotRight:
				mode[n1] = 2
				# print "forward,right"
			elif inNegTrans and inRotLeft:
				mode[n1] = 3
				# print "back, left"
			elif inNegTrans and inRotRight:
				mode[n1] = 4
				# print "back, right"
			elif inPosTrans:
				mode[n1] = 5
				# print "forward"
			elif inNegTrans:
				mode[n1] = 6
				# print "backward"
			elif inRotLeft:
				mode[n1] = 7
				# print "left"
			elif inRotRight:
				mode[n1] = 8
				# print "right"
			if (listAL1[j][num]/listAL2[j][num] > 0.6):
				# if np.sqrt((COM_x1[j][num]-q4[1])**2/((halfWidth-5.)**2) + (COM_y1[j][num]-q4[0])**2/((halfHeight-5.)**2)) > 1:
				modeQuality[n1] = 1
				print "Frame", n1, " flagged. Insufficient pixel information."

			if ((COM_x1[j][num]-q4[1])**2/((halfWidth-7.)**2) + (COM_y1[j][num]-q4[0])**2/((halfHeight -7.)**2)) > 1:
				onWall[n1] = 1
			else: 
				onWall[n1]=0
			nearTout= isNearTempBarrier(COM_y1[j][num],COM_x1[j][num],denoisedThetas[j][num],num)
			if nearTout[0]:
				nearTempB[n1]= nearTout[1] 
				nearTempB_dist[n1]=nearTout[2]
			n1+=1


	#         im5 = vid.get_data(n1)#for demoing. 
	#         print mode[n1-1]
	#         # cTheta = mode
	#         # p3 = np.round((100-10*np.sin(np.pi*cTheta/180),100+10*np.cos(np.pi*cTheta/180)))
	#         imageG5 = cv2.cvtColor(im5,cv2.COLOR_BGR2GRAY)
	#         imageG5_1 = imageG5#[25:200,25:200]
	#         cv2.namedWindow("2",cv2.WINDOW_NORMAL)
	#         # cv2.circle(imageG5_1,tuple(q4),1,(0,255,0),-2)
	#         # cv2.circle(imageG5_1,tuple(q4),75,(0,255,0),1)
	#         # cv2.line(imageG5_1,(q4[0]-80,q4[1]),(q4[0]+80,q4[1]),(0,255,0),1)
	#         # cv2.line(imageG5_1,(q4[0],q4[1]-80),(q4[0],q4[1]+80),(0,255,0),1)
	#         # cv2.line(imageG5_1,(100,100),tuple(p3.astype(int)),(0,0,255),1)
	#         cv2.imshow("2",imageG5_1)
	#         cv2.resizeWindow("2",500,500)
	#         cv2.waitKey(60)
	# cv2.destroyAllWindows()
	return (mode,modeQuality,onWall,nearTempB,nearTempB_dist)
import copy
def eventTabulate(mode,nStart,nFinish,onWall,quad,tempB1,tempB1_dist,quad_h,quad_b):
	#converts the list of behavioral modes into a short list of behaviors. 
	#for example, a list of forward moves, left turns, and moving left turns
	#would be grouped into a single event, but as soon as a backward move,
	# a rest, or a rightward move begins,  a new event would be initiated. 
	cMode = -1
	forward_left = [1,7]
	forward_right = [2,8]
	backward_left = [3,7]
	backward_right = [4,8]
	rest = [0]
	forward = [5]
	backward = [6]
	listOfMoveSets = [set(forward_left),set(forward_right),set(backward_left),set(backward_right),set(rest),set(forward),set(backward)]
	cEvent = simpleBehaviorEvent(0)
	allEvents = []
	nums = xrange(0,nFinish-nStart)
	vid = imageio.get_reader(filename,  'ffmpeg')
	vDims =vid.get_meta_data()['source_size'][::-1]
	maxProj = np.zeros(vDims)
	for num in nums:
		# if num >500 and num<1000:
		# 	im5_1 = vid.get_data(num)
		# 	cv2.imshow("masked",im5_1)
		# 	cv2.resizeWindow('masked', 600,600)
		# 	cv2.waitKey(50)
		if modeQuality[num] == 1:
			cEvent.markQualityIsBad()

		if onWall[num] == 1:
			cEvent.markOnWall()

		if tempB1[num] !=-180:
			cEvent.markNearTempBarrier(tempB1[num],num)
		cEvent.minTempB_Dist(tempB1_dist[num])

		cEvent.addQuadrant(int(quad[num]))
		cEvent.addQuadrant_head(int(quad_h[num]))
		cEvent.addQuadrant_back(int(quad_b[num]))
		cMode = mode[num]
		#print cMode
		pastModes = cEvent.getModes()
		if cMode not in set(pastModes):
			pastModes = np.append(pastModes,cMode)
			newMode1 = False
			for i in xrange(0,len(listOfMoveSets)):
				#if we no longer fit into one of these move categories, make a new event. 
				newMode = all([pastModes[l] in listOfMoveSets[i] for l in xrange(0,len(pastModes))])
				if newMode == True: 
					newMode1 = True
			if newMode1 == True:
				cEvent.addModeType(cMode)
			else:
				cEvent.endEvent(num-1)
				s11,f11 = cEvent.getStartAndFinish()
				##check for local velocity minima... if we find them, break the event into pieces. 
				a11 = fAngs_deriv[s11:f11]
				if len(a11)>10 and s11<f11 and np.sum(np.abs(a11))>0:
					indexes = np.unique(peakutils.indexes(-np.abs(a11),thres=0.25,min_dist=10))
				else:
					indexes=[]
				if np.all(np.abs(a11)>1.5) and len(indexes)>0 and (f11-s11)>10:
					tempEvent = copy.copy(cEvent)
					# cEvent.frameEnd = f11
					# for j in xrange(s11,f11+1):
					# 	im5_1 = vid.get_data(j)
					# 	maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
					# cEvent.addMaxProj(maxProj)
					# allEvents.append(cEvent)
					# maxProj = np.zeros(vDims)
					if indexes[0]<5:
						indexes = np.delete(indexes,0)
					if len(indexes)>1 and indexes[len(indexes)-1]>(f11-s11)-5 and f11>s11:
						indexes = np.delete(indexes,len(indexes)-1)
					if len(indexes)>0:
						for h in xrange(0,len(indexes)+1):
							# print tempEvent.frameStart,tempEvent.frameEnd,s11,f11
							if h<len(indexes):
								if indexes[h]>5 and indexes[h]<(len(a11)-5):
									cEvent = copy.copy(tempEvent)

									if h==0:
										cEvent.frameStart=s11
										cEvent.frameEnd=indexes[h]+s11-1
									elif h>0 and h < len(indexes):
										cEvent.frameStart=indexes[h-1]+s11
										cEvent.frameEnd = indexes[h]+s11-1
									# print tempEvent.frameStart,tempEvent.frameEnd,s11,f11
									for j in xrange(cEvent.frameStart,cEvent.frameEnd+1):
										im5_1 = vid.get_data(j)
										maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
									cEvent.addMaxProj(maxProj)
									# print cEvent.frameStart,cEvent.frameEnd,f11-s11,indexes[h],tempEvent.frameStart,tempEvent.frameEnd
									allEvents.append(cEvent)
									maxProj = np.zeros(vDims)
							else:
								if indexes[h-1]>5 and indexes[h-1]<(len(a11)-5):
									cEvent = copy.copy(tempEvent)

									cEvent.frameStart = indexes[h-1]+s11
									cEvent.frameEnd = f11
										
									for j in xrange(cEvent.frameStart,cEvent.frameEnd+1):
										im5_1 = vid.get_data(j)
										maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
									cEvent.addMaxProj(maxProj)
									allEvents.append(cEvent)
									maxProj = np.zeros(vDims)
								else:
									if h>0 and cEvent.frameEnd!=f11:
										cEvent.frameStart = cEvent.frameEnd
										cEvent.frameEnd = f11
									else:
										cEvent = copy.copy(tempEvent)
									for j in xrange(cEvent.frameStart,cEvent.frameEnd+1):
										im5_1 = vid.get_data(j)
										maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
									cEvent.addMaxProj(maxProj)
									allEvents.append(cEvent)
									maxProj = np.zeros(vDims)
					else:
						for j in xrange(s11,f11+1):
							im5_1 = vid.get_data(j)
							maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
						cEvent.addMaxProj(maxProj)
						allEvents.append(cEvent)
						maxProj = np.zeros(vDims)
					# print f11,num
					cEvent = simpleBehaviorEvent(num)
					cEvent.addModeType(cMode)
					# plt.subplot(121)
					# plt.plot(a11)
					# plt.plot(indexes,a11[indexes],"bo")
					# plt.show()
				else:
					for j in xrange(s11,f11+1):
						im5_1 = vid.get_data(j)
						maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
					cEvent.addMaxProj(maxProj)
					allEvents.append(cEvent)
					maxProj = np.zeros(vDims)
					cEvent = simpleBehaviorEvent(num)
					cEvent.addModeType(cMode)
	cEvent.endEvent(num)
	for j in xrange(s11,f11+1):
		im5_1 = vid.get_data(j)
		maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
	cEvent.addMaxProj(maxProj)
	allEvents.append(cEvent)
	return allEvents

def eventClassify_withLength(allEvents,COM_x1,COM_y1):
	#for calculating stats about the events produced in eventTabulate. 
	#This produces the number of frames spent in each mode type. 
	eventType = np.zeros(9)
	for j in xrange(0,len(allEvents)):
		cM = allEvents[j].getModes()
		[s1,f1] = allEvents[j].getStartAndFinish()
		fCount = f1-s1
		if allEvents[j].quality <5:
			if 0 in set(cM):
				eventType[0]+=fCount#rest
			elif 1 in set(cM):
				eventType[1]+=fCount#forward, left
			elif 2 in set(cM):
				eventType[2]+=fCount #forward,right
			elif 3 in set(cM):
				eventType[3]+=fCount#back,left
			elif 4 in set(cM):
				eventType[4]+=fCount#back,right
			elif 7 in set(cM):
				eventType[7]+=fCount#stationary left
			elif 8 in set(cM):
				eventType[8]+=fCount #stationary right 
			elif 5 in set(cM):
				eventType[5]+=fCount#forward walk
			elif 6 in set(cM):
				eventType[6]+=fCount#backward walk

	return eventType

def eventClassify(allEvents,COM_x1,COM_y1):
	#for calculating stats about the events produced in eventTabulate. 
	#This produces the number of occurrences of each mode type. 
	eventType = np.zeros(9)
	for j in xrange(0,len(allEvents)):
		cM = allEvents[j].getModes()
		if allEvents[j].quality <5:
			if 0 in set(cM):
				eventType[0]+=1#rest
			elif 1 in set(cM):
				eventType[1]+=1#forward, left
			elif 2 in set(cM):
				eventType[2]+=1 #forward,right
			elif 3 in set(cM):
				eventType[3]+=1#back,left
			elif 4 in set(cM):
				eventType[4]+=1 #back,right
			elif 7 in set(cM):
				eventType[7]+=1 #stationary left
			elif 8 in set(cM):
				eventType[8]+=1 #stationary right 
			elif 5 in set(cM):
				eventType[5]+=1#forward walk
			elif 6 in set(cM):
				eventType[6]+=1 #backward walk
	return eventType
=======
    # function to calculate a moving average of a trajectory.
    # Note: Should only be used with odd window sizes.

    # if window_size%2==0:
    # 	print('window size is even for movingaverage fctn')
    # 	quit()

    if np.size(interval) > 0:
        window = np.ones(int(window_size))
        averaged = np.convolve(interval, window, 'same')/float(window_size)
        win1 = int((window_size-1)/2)
        for j in range(0, win1):
            # print j
            averaged[j] = np.sum(interval[0:2*j+1])/(2.*j + 1.)
        for j in range(-win1-1, 0):
            # print j,averaged[j:]
            averaged[j] = np.sum(interval[2*(j+1)-1:])/(-2.*(j+1)+1)
    else:
        averaged = []
    return averaged


def coreDataExtract(filename):
    # this function performs the data extraction from the video, converting the fly to a binary image
    # and then performing a least squares ellipse fit to find the orientation of the fly.
    # A few different centroid calculations can be done. At the moment, the centroid is taken as the
    # center of mass of the fly. This will be updated soon however.
    import imageio
    import cv2
    vid = imageio.get_reader(filename, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    vDims = vid.get_meta_data()['source_size']
    vLength = vid.count_frames()  # vid.get_length deprecated
    nStart = 0
    nFinish = vLength
    nums = range(nStart, nFinish)

    allThetas = np.array([])
    allCenters = np.array([])
    allCOM = np.array([])
    al1s = np.array([])
    al2s = np.array([])
    listThetas = []
    listCenters = []
    listCOM = []
    listAL1 = []
    listAL2 = []
    # cDir = theta
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frames = []
    badCount = 0
    # print vDims
    # print kernel
    mask = np.ones(vDims, dtype="uint8")*255
    for i in range(0, vDims[0]):
        for j in range(0, vDims[1]):
            if np.sqrt((q4[0]-i)**2 + (q4[1]-j)**2) > ((halfWidth + halfHeight)/2+1):
                mask[i, j] = 0

    a1, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # mask.tolist()
    for num in nums:
        im5 = vid.get_data(num)
        # if num ==0:
        # 	plt.imshow(im5)
        imageG5_1 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
        # imageG5_1 = cv2.bitwise_and(imageG5,imageG5,mask=mask)#[25:20*scaling/4.70,25:20*scaling/4.70]
        r1, t5_1 = cv2.threshold(imageG5_1, threshVal1, 255, cv2.THRESH_BINARY)
        # tOld = t5_1
        # [25:20*scaling/4.70,25:20*scaling/4.70]
        t5_1 = cv2.bitwise_and(mask.T, t5_1, t5_1)

        t5_1 = cv2.erode(t5_1, kernel, iterations=1)
        t5_1 = cv2.dilate(t5_1, kernel, iterations=1)
        tOld = t5_1
        t5_1 = cv2.bitwise_not(t5_1)
        contours, hierarchy = cv2.findContours(t5_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # only record if we have sufficient data to determine location/orientation
        if len(contours) < 2:
            if 1-suppress:
                print(num, "problem frame:insufficient data")
            if len(allThetas) > 0:
                listThetas.append(allThetas)
                listCenters.append(allCenters)
                listCOM.append(allCOM)
                listAL1.append(al1s)
                listAL2.append(al2s)
                # print len(allThetas), len(al1s)
                allThetas = np.array([])
                allCenters = np.array([])
                allCOM = np.array([])
                al1s = np.array([])
                al2s = np.array([])
                badCount = 0
                frames.append(num)
        else:
            cnt = contours[1]
            # if low information, don't update postition/orientation.
            if len(cnt) < 5:
                if np.size(allThetas) > 0:
                    if len(frames) % 2 == 0:
                        frames.append(num)
                    allThetas = np.append(allThetas, theta)
                    allCenters = np.append(allCenters, [cx, cy])
                    allCOM = np.append(allCOM, cCOM)
                    al1s = np.append(al1s, al1)
                    al2s = np.append(al2s, al2)
            else:
                ellipse = cv2.fitEllipse(cnt)
                cx, cy = ellipse[0]
                al1, al2 = ellipse[1]
                theta = ellipse[2]
                if centerType == 'COM':
                    cCOM = findCOM(tOld, vDims)
                else:
                    cCOM = [cy, cx]
                # cCOM1 = [cCOM[1],cCOM[0]]
                # t5_1 = cv2.circle(t5_1,tuple([int(i1) for i1 in cCOM1]),1,0,-1)
                # cv2.imshow("masked",t5_1)
                # cv2.resizeWindow('masked', 600,600)
                # cv2.waitKey(50)
                # (np.sqrt(((cCOM[0]-q4[0])/(halfWidth-10))**2 + ((cCOM[1]-q4[1])/(halfHeight-10)**2)) > 1):
                if (al1/al2 > 0.75):
                    if badCount > 1:
                        listThetas.append(allThetas)
                        listCenters.append(allCenters)
                        listCOM.append(allCOM)
                        listAL1.append(al1s)
                        listAL2.append(al2s)
                        allThetas = np.array([])
                        allCenters = np.array([])
                        allCOM = np.array([])
                        al1s = np.array([])
                        al2s = np.array([])
                        frames.append(num)
                        badCount = 0
                    elif np.size(allThetas) == 0:
                        badCount = 0
                        # print "still in bad region",num
                        # if len(frames)%2 ==1:
                        #     frames = []
                    else:
                        if len(frames) % 2 == 0:
                            frames.append(num)
                        if num == nStart:
                            theta_old = theta

                        if centerType == 'COM':
                            allCOM = np.append(allCOM, cCOM)
                        else:
                            allCOM = np.append(allCOM, [cy, cx])
                        # allThetas = np.append(allThetas,theta)
                        allCenters = np.append(allCenters, [cx, cy])
                        al1s = np.append(al1s, al1)
                        al2s = np.append(al2s, al2)
                        allThetas = np.append(allThetas, theta_old)
                        # print 'bc',badCount
                        badCount += 1

                else:
                    # if len(frames)%2 ==0:
                    #     frames.append(num)
                    if np.size(allThetas) == 0:
                        frames.append(num)
                    if centerType == 'COM':
                        allCOM = np.append(allCOM, cCOM)
                    else:
                        allCOM = np.append(allCOM, [cy, cx])
                    # allThetas = np.append(allThetas,theta)
                    allCenters = np.append(allCenters, [cx, cy])
                    al1s = np.append(al1s, al1)
                    al2s = np.append(al2s, al2)
                    allThetas = np.append(allThetas, theta)
                    theta_old = theta

    listThetas.append(allThetas)
    listCenters.append(allCenters)
    listCOM.append(allCOM)
    listAL1.append(al1s)
    listAL2.append(al2s)
    if len(frames) % 2 == 1:
        frames.append(num+1)
    COM_x1 = []
    COM_y1 = []
    vx_1 = []
    vy_1 = []
    denoisedThetas = []
    denoisedThetas_deriv = []
    # stitching "off-wall"/usable sections together
    for j in range(min(len(listThetas), int(len(frames)/2))):
        if len(listThetas[j]) < 9:
            k1 = int(np.floor(len(listThetas[j])/2))+1
            k2 = k1
        else:
            k1 = 9
            k2 = 7
        if 1-suppress:
            print(k1, '\n')

        COM_y1.append(movingaverage(listCOM[j][1::2], k1))
        COM_x1.append(movingaverage(listCOM[j][0::2], k1))
        if k1 == 1:
            vx_1.append(np.zeros(1))
            vy_1.append(np.zeros(1))
        else:
            vx_1.append(movingaverage(np.gradient(COM_x1[j]), k2))
            vy_1.append(movingaverage(np.gradient(COM_y1[j]), k2))
        lStart = frames[2*j]
        lFinish = frames[2*j+1]
        numsMinusZero = range(1, lFinish-lStart)
        allDiffs = np.zeros(lFinish-lStart)
        allDiffs[0] = listThetas[j][0]
        for num in numsMinusZero:
            theta = listThetas[j][num]
            cDir = listThetas[j][num-1]
            mi2 = np.argmin(
                [abs(theta-cDir), abs(180+theta-cDir), abs(-180+theta-cDir)])

            change = np.min(
                [abs(theta-cDir), abs(180+theta-cDir), abs(-180+theta-cDir)])
            bla = [(theta-cDir), (180+theta-cDir), (-180+theta-cDir)]
            allDiffs[num] = allDiffs[num-1] + change*np.sign(bla[mi2])

        denoisedThetas.append(movingaverage(allDiffs, k2))
        if k1 == 1:
            denoisedThetas_deriv.append(np.zeros(1))
        else:
            denoisedThetas_deriv.append(movingaverage(
                np.gradient(movingaverage(allDiffs, k2)), k2))

    # now calculate some statistics about hot/cold regions
    quad = []
    qH = []
    qB = []
    for j in range(min(len(listThetas), int(len(frames)/2))):
        lStart = frames[2*j]
        lFinish = frames[2*j+1]
        nums121 = range(0, lFinish-lStart)
        quad_1 = np.zeros(lFinish-lStart)
        quad_head = np.zeros(lFinish-lStart)
        quad_back = np.zeros(lFinish-lStart)

        for ind in nums121:
            quad_1[ind] = calcQuadrant(COM_y1[j][ind], COM_x1[j][ind])
            headPos = headPosition(
                COM_y1[j][ind], COM_x1[j][ind], denoisedThetas[j][ind])
            backPos = backPosition(
                COM_y1[j][ind], COM_x1[j][ind], denoisedThetas[j][ind])

            # plt.plot(headPos[0],headPos[1],'ro')
            # plt.plot(backPos[0],backPos[1],'bo')
            # plt.plot(COM_y1[j][ind],COM_x1[j][ind],'go')
            # plt.show()

            quad_head[ind] = calcQuadrant(headPos[0], headPos[1])
            quad_back[ind] = calcQuadrant(backPos[0], backPos[1])

        quad.append(quad_1)
        qH.append(quad_head)
        qB.append(quad_back)

    return (COM_x1, COM_y1, vx_1, vy_1, denoisedThetas, denoisedThetas_deriv, nStart, nFinish, listAL1, listAL2, frames, quad, qH, qB)


def headPosition(x_p, y_p, theta_p):
    # calculates head position
    # np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
    cToHead = np.array([np.sin(np.pi/180.*theta_p),
                       np.cos(np.pi/180.*theta_p)])
    cVector = np.array([x_p, y_p])+hBL*cToHead
    return cVector


def backPosition(x_p, y_p, theta_p):
    # calculates back(opposite of head) position
    # np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
    cToHead = np.array([np.sin(np.pi/180.*theta_p),
                       np.cos(np.pi/180.*theta_p)])
    cVector = np.array([x_p, y_p])-hBL*cToHead
    return cVector


class simpleBehaviorEvent:
    # elementary behavioral event class.
    def __init__(self, fStart):
        self.frameStart = fStart
        self.modes = []
        self.quality = 0
        self.onWall = 0
        self.distTraveled = 0.0
        self.quadrants = []
        self.quadrants_head = []
        self.quadrants_back = []
        self.nearTempBarrier = -180
        self.minDistTempB = 1000
        self.arenaFile = arenaFile
        self.showQuadrants = showQuadrants
        self.frameEntryT = -1

    def addModeType(self, mode):
        self.modes = np.append(self.modes, mode)

    def getModes(self):
        return self.modes

    def getStartAndFinish(self):
        return [self.frameStart, self.frameEnd]

    def markQualityIsBad(self):
        self.quality += 1

    def markOnWall(self):
        self.onWall = 1

    def markNearTempBarrier(self, ang, fNum):
        if self.nearTempBarrier == -180:
            self.nearTempBarrier = ang
            self.frameEntryT = fNum

    def minTempB_Dist(self, cDist):
        if cDist < self.minDistTempB:
            self.minDistTempB = cDist

    def addQuadrant(self, qNum):
        if qNum not in self.quadrants:
            self.quadrants.append(qNum)

    def addQuadrant_head(self, qNum):
        if qNum not in self.quadrants_head:
            self.quadrants_head.append(qNum)

    def addQuadrant_back(self, qNum):
        if qNum not in self.quadrants_back:
            self.quadrants_back.append(qNum)

    def addMaxProj(self, maxProj):
        buff = io.BytesIO()  # stringIO deprecated
        if self.frameEnd-self.frameStart >= 0:
            img1 = Image.fromarray(maxProj/(self.frameEnd-self.frameStart + 1))
        else:
            img1 = Image.fromarray(maxProj/(self.frameEnd-self.frameStart + 2))
        img1 = img1.convert('RGB')
        img1.save(buff, "JPEG", quality=100)
        self.maxProj = buff

    def endEvent(self, fEnd):
        self.frameEnd = fEnd


def decomposeVelocity(vx_1, vy_1, denoisedThetas):
    # projects the translational velocity onto a new coordinate system
    # where rather than x and y velocity, we have forward/backward and left/right slip velocity
    transV = []
    slipV = []
    for j in range(0, len(vx_1)):
        listLength = len(vx_1[j])
        nums2 = range(0, listLength)
        # print listLength,len(vx_1),j
        transV.append(np.zeros(listLength))
        slipV.append(np.zeros(listLength))
        # print listLength, len(denoisedThetas[j])
        for num in nums2:
            # print listLength,len(vx_1),j,num,len(denoisedThetas[j])
            trueV = np.array([vy_1[j][num], vx_1[j][num]])
            cDir = np.array([-np.sin(np.pi*denoisedThetas[j][num]/180.),
                            np.cos(np.pi*denoisedThetas[j][num]/180.)])
            cDir_tang = np.array([np.cos(
                np.pi*denoisedThetas[j][num]/180.), np.sin(np.pi*denoisedThetas[j][num]/180.)])
            transV[j][num] = np.dot(trueV, cDir)
            slipV[j][num] = np.dot(trueV, cDir_tang)
        # plt.plot(transV[j])
        # plt.plot(slipV[j])
        # plt.show()
    return (transV, slipV)


def checkOrientation(transV, denoisedThetas1):
    redo = 0
    for j in range(0, len(transV)):
        # or (previousTheta!=-1000 and np.abs(previousTheta-denoisedThetas1[j][0])>150):
        if np.sum(transV[j]) < 0 or np.sum([tV1 < -0.01 for tV1 in transV[j]]) > np.sum([tV1 > 0.01 for tV1 in transV[j]]):
            denoisedThetas1[j] = [dT + 180 for dT in denoisedThetas1[j]]
            redo = 1

    return (redo, denoisedThetas1)


def timepointClassify(transV, denoisedThetas_deriv, listAL1, listAL2, COM_x1, COM_y1, frames):
    # this function converts translational/rotational information into an behavioral mode for that frame.
    # the output of this is fed into the eventTabulate function to form "behavioral" events.
    transThresh = 0.15*scaling/4.7
    slipThresh = 0.15*scaling/4.7
    rotThresh = 1.5
    mode = np.ones(nFinish-nStart)*-1
    modeQuality = np.zeros(nFinish-nStart)
    onWall = np.ones(nFinish-nStart)*1
    nearTempB = np.ones(nFinish-nStart)*-180.
    nearTempB_dist = np.ones(nFinish-nStart)*-1.
    #q4 = [99,95]
    p_max = 5
    coordInfo = np.ones((nFinish-nStart, 4))*-1
    # vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.
    print("===List of modes occuring in every frame===")
    # now, let's classify our "events". these can be left and right turns(stationary or directed),forward and backward movements, or rests.
    for j in range(0, len(transV)):
        listLength = len(transV[j])
        nums2 = range(0, listLength)
        n1 = frames[2*j]
        for num in nums2:
            [inPosTrans, inNegTrans, inSlip, inRotRight, inRotLeft] = [0, 0, 0, 0, 0]
            if transV[j][num] > transThresh:
                inPosTrans = 1
            if transV[j][num] < -transThresh:
                inNegTrans = 1
            if abs(slipV[j][num]) > slipThresh:
                inSlip = 1
            if denoisedThetas_deriv[j][num] > rotThresh:
                inRotRight = 1
            if denoisedThetas_deriv[j][num] < -rotThresh:
                inRotLeft = 1
            coordInfo[n1, :] = [inPosTrans, inNegTrans, inRotLeft, inRotRight]
            n1 += 1
    # now extend left/right limits of events if possible.
    posT = coordInfo[:, 0]
    negT = coordInfo[:, 1]
    leftR = coordInfo[:, 2]
    rightR = coordInfo[:, 3]
    for j in range(1, len(mode)):
        # forward motion check
        if (posT[j]-posT[j-1]) == 1:
            pushLeft = 1
            notFlat = 1
            p = 1
            while pushLeft and (j-p) >= 0 and p < p_max:
                sameSign = (tV[j-p] - tV[j-p+1]) < 0  # should be decreasing
                notFlat = (posT[j-p] == 0)
                if tV[j-p] > 0 and notFlat and sameSign:
                    posT[j-p] = 1
                elif not notFlat:
                    pushLeft = 0
                    posT[j-p] = 0
                else:
                    pushLeft = 0

                p += 1

        elif(posT[j]-posT[j-1]) == -1:
            pushRight = 1
            notFlat = 1
            p = 0
            while pushRight and (j+p) < len(mode) and p < p_max:
                sameSign = (tV[j+p] - tV[j+p-1]) < 0  # should be decreasing
                notFlat = (posT[j+p] == 0)
                if tV[j+p] > 0 and notFlat and sameSign:
                    posT[j+p] = 1
                elif not notFlat:
                    pushRight = 0
                    posT[j+p] = 0
                else:
                    pushRight = 0
                p += 1
        # backward motion
    for j in range(1, len(mode)):
        if (negT[j]-negT[j-1]) == 1:
            pushLeft = 1
            notFlat = 1
            p = 1
            while pushLeft and (j-p) >= 0 and p < p_max:
                sameSign = (tV[j-p] - tV[j-p+1]) > 0  # should be increasing
                notFlat = (negT[j-p] == 0)
                if tV[j-p] < 0 and notFlat and sameSign:
                    negT[j-p] = 1
                elif not notFlat:
                    pushLeft = 0
                    negT[j-p] = 0
                else:
                    pushLeft = 0
                p += 1

        elif(negT[j]-negT[j-1]) == -1:
            pushRight = 1
            notFlat = 1
            p = 0
            while pushRight and (j+p) < len(mode) and p < p_max:
                sameSign = (tV[j+p] - tV[j+p-1]) > 0  # should be increasing
                notFlat = (negT[j+p] == 0)
                if tV[j+p] < 0 and notFlat and sameSign:
                    negT[j+p] = 1
                elif not notFlat:
                    pushRight = 0
                    negT[j+p] = 0
                else:
                    pushRight = 0
                p += 1

        # left turning motion
    for j in range(1, len(mode)):
        if (leftR[j]-leftR[j-1]) == 1:
            pushLeft = 1
            notFlat = 1
            p = 1
            # leftR[j-1] = 1
            while pushLeft and (j-p) >= 0 and p < p_max:
                # should be increasing
                sameSign = (fAngs_deriv[j-p] - fAngs_deriv[j-p+1]) > 0
                notFlat = (leftR[j-p] == 0)
                if fAngs_deriv[j-p] < 0 and notFlat and sameSign:
                    leftR[j-p] = 1
                elif not notFlat:
                    pushLeft = 0
                    leftR[j-p] = 0
                else:
                    pushLeft = 0
                p += 1
            # print "goingleft",sameSign,notFlat,j,j-p,leftR[j-p]

        elif(leftR[j]-leftR[j-1]) == -1:
            pushRight = 1
            notFlat = 1
            p = 0
            while pushRight and (j+p) < len(mode) and p < p_max:
                # should be increasing
                sameSign = (fAngs_deriv[j+p] - fAngs_deriv[j+p-1]) > 0
                notFlat = (leftR[j+p] == 0)
                if fAngs_deriv[j+p] < 0 and notFlat and sameSign:
                    leftR[j+p] = 1
                elif not notFlat:
                    pushRight = 0
                    leftR[j+p] = 0
                else:
                    pushRight = 0
                # print sameSign,notFlat,j,leftR[j],j+p,leftR[j+p],(fAngs_deriv[j+p] - fAngs_deriv[j+p-1])
                p += 1
        # 		# print sameSign,notFlat,j+p,leftR[j+p]
        # if j ==195:
        # 	asdfa

        # right turning motion
    for j in range(1, len(mode)):
        if (rightR[j]-rightR[j-1]) == 1:
            pushLeft = 1
            notFlat = 1
            p = 1
            while pushLeft and (j-p) >= 0 and p < p_max:
                # should be decreasing
                sameSign = (fAngs_deriv[j-p] - fAngs_deriv[j-p+1]) < 0
                notFlat = (leftR[j-p] == 0)
                if fAngs_deriv[j-p] > 0 and notFlat and sameSign:
                    rightR[j-p] = 1
                elif not notFlat:
                    pushLeft = 0
                    rightR[j-p] = 0
                else:
                    pushLeft = 0
                p += 1

        elif(rightR[j]-rightR[j-1]) == -1:
            pushRight = 1
            notFlat = 1
            p = 0
            while pushRight and (j+p) < len(mode) and p < p_max:
                # should be decreasing
                sameSign = (fAngs_deriv[j+p] - fAngs_deriv[j+p-1]) < 0
                notFlat = (leftR[j+p] == 0)
                if fAngs_deriv[j+p] > 0 and notFlat and sameSign:
                    rightR[j+p] = 1
                elif not notFlat:
                    pushRight = 0
                    rightR[j+p] = 0
                else:
                    pushRight = 0
                p += 1
    # now use extended data for for mode determination.
    coordInfo = np.array([posT, negT, rightR, leftR]).T
    for j in range(0, len(transV)):
        listLength = len(transV[j])
        nums2 = range(0, listLength)
        n1 = frames[2*j]
        for num in nums2:
            [inPosTrans, inNegTrans, inRotRight, inRotLeft] = coordInfo[n1, :]
            dsum = np.sum([inPosTrans, inNegTrans, inRotRight, inRotLeft])
            # print inPosTrans,inNegTrans,inRotRight,inRotLeft,n1
            # if n1 == 195:
            # 	asdfasf
            if dsum == 0:
                inRest = 1
                mode[n1] = 0
                # print "rest"
            elif inPosTrans and inRotLeft:
                mode[n1] = 1
                # print "forward, left"
            elif inPosTrans and inRotRight:
                mode[n1] = 2
                # print "forward,right"
            elif inNegTrans and inRotLeft:
                mode[n1] = 3
                # print "back, left"
            elif inNegTrans and inRotRight:
                mode[n1] = 4
                # print "back, right"
            elif inPosTrans:
                mode[n1] = 5
                # print "forward"
            elif inNegTrans:
                mode[n1] = 6
                # print "backward"
            elif inRotLeft:
                mode[n1] = 7
                # print "left"
            elif inRotRight:
                mode[n1] = 8
                # print "right"
            if (listAL1[j][num]/listAL2[j][num] > 0.6):
                # if np.sqrt((COM_x1[j][num]-q4[1])**2/((halfWidth-5.)**2) + (COM_y1[j][num]-q4[0])**2/((halfHeight-5.)**2)) > 1:
                modeQuality[n1] = 1
                if 1-suppress:
                    print("Frame", n1, " flagged. Insufficient pixel information.")

            if ((COM_x1[j][num]-q4[1])**2 + (COM_y1[j][num]-q4[0])**2) > (halfWidth-hBL)**2:
                onWall[n1] = 1
            else:
                onWall[n1] = 0

            qHot1 = isQuadHot(calcQuadrant(COM_y1[j][num]-hBL*np.sin(np.pi/180.*denoisedThetas[j][num]),
                              COM_x1[j][num]+hBL*np.cos(np.pi/180*denoisedThetas[j][num])), showQuadrants)
            qHot_rear1 = isQuadHot(calcQuadrant(COM_y1[j][num]+hBL*np.sin(
                np.pi/180.*denoisedThetas[j][num]), COM_x1[j][num]-hBL*np.cos(np.pi/180*denoisedThetas[j][num])), showQuadrants)
            nearTout = isNearTempBarrier(
                COM_y1[j][num], COM_x1[j][num], denoisedThetas[j][num], num, qHot1, qHot_rear1)
            if nearTout[0]:
                nearTempB[n1] = nearTout[1]
                nearTempB_dist[n1] = nearTout[2]
            n1 += 1

    #         im5 = vid.get_data(n1)#for demoing.
    #         print mode[n1-1]
    #         # cTheta = mode
    #         # p3 = np.round((100-10*np.sin(np.pi*cTheta/180),100+10*np.cos(np.pi*cTheta/180)))
    #         imageG5 = cv2.cvtColor(im5,cv2.COLOR_BGR2GRAY)
    #         imageG5_1 = imageG5#[25:20*scaling/4.70,25:20*scaling/4.70]
    #         cv2.namedWindow("2",cv2.WINDOW_NORMAL)
    #         # cv2.circle(imageG5_1,tuple(q4),1,(0,255,0),-2)
    #         # cv2.circle(imageG5_1,tuple(q4),75,(0,255,0),1)
    #         # cv2.line(imageG5_1,(q4[0]-80,q4[1]),(q4[0]+80,q4[1]),(0,255,0),1)
    #         # cv2.line(imageG5_1,(q4[0],q4[1]-80),(q4[0],q4[1]+80),(0,255,0),1)
    #         # cv2.line(imageG5_1,(100,100),tuple(p3.astype(int)),(0,0,255),1)
    #         cv2.imshow("2",imageG5_1)
    #         cv2.resizeWindow("2",500,500)
    #         cv2.waitKey(60)
    # cv2.destroyAllWindows()
    return (mode, modeQuality, onWall, nearTempB, nearTempB_dist)


def eventTabulate(mode, nStart, nFinish, onWall, quad, tempB1, tempB1_dist, quad_h, quad_b):
    # converts the list of behavioral modes into a short list of behaviors.
    # for example, a list of forward moves, left turns, and moving left turns
    # would be grouped into a single event, but as soon as a backward move,
    # a rest, or a rightward move begins,  a new event would be initiated.
    cMode = -1
    allTurns = set([1, 2, 3, 4, 7, 8])
    forward_left = [1, 7]
    forward_right = [2, 8]
    backward_left = [3, 7]
    backward_right = [4, 8]
    rest = [0]
    forward = [5]
    backward = [6]
    listOfMoveSets = [set(forward_left), set(forward_right), set(
        backward_left), set(backward_right), set(rest), set(forward), set(backward)]
    cEvent = simpleBehaviorEvent(0)
    allEvents = []
    nums = range(0, nFinish-nStart)
    vid = imageio.get_reader(filename,  'ffmpeg')
    vDims = vid.get_meta_data()['source_size'][::-1]
    maxProj = np.zeros(vDims)
    for num in nums:
        # if num >500 and num<1000:
        # 	im5_1 = vid.get_data(num)
        # 	cv2.imshow("masked",im5_1)
        # 	cv2.resizeWindow('masked', 600,600)
        # 	cv2.waitKey(50)
        if modeQuality[num] == 1:
            cEvent.markQualityIsBad()

        if onWall[num] == 1:
            cEvent.markOnWall()

        if tempB1[num] != -180:
            cEvent.markNearTempBarrier(tempB1[num], num)
        cEvent.minTempB_Dist(tempB1_dist[num])

        cEvent.addQuadrant(int(quad[num]))
        cEvent.addQuadrant_head(int(quad_h[num]))
        cEvent.addQuadrant_back(int(quad_b[num]))
        cMode = mode[num]
        # print cMode
        pastModes = cEvent.getModes()
        if cMode not in set(pastModes):
            pastModes = np.append(pastModes, cMode)
            newMode1 = False
            for i in range(0, len(listOfMoveSets)):
                # if we no longer fit into one of these move categories, make a new event.
                newMode = all([pastModes[l] in listOfMoveSets[i]
                              for l in range(0, len(pastModes))])
                if newMode == True:
                    newMode1 = True
            if newMode1 == True:
                cEvent.addModeType(cMode)
            else:
                cEvent.endEvent(num-1)
                s11, f11 = cEvent.getStartAndFinish()
                # check for local velocity minima... if we find them, break the event into pieces.
                a11 = fAngs_deriv[s11:f11+1]
                cModes = cEvent.getModes()

                snipCheck = np.sum(
                    [cModes[i] in allTurns for i in range(0, len(cModes))]) > 0
                if len(a11) > 10 and s11 < f11 and np.sum(np.abs(a11)) > 0 and snipCheck:
                    indexes = np.unique(
                        peakutils.indexes(-np.abs(a11), thres=0.2, min_dist=6))
                else:
                    indexes = []
                if len(indexes) > 0 and (f11-s11) > 10 and snipCheck:
                    tempEvent = copy.copy(cEvent)
                    # cEvent.frameEnd = f11
                    # for j in range(s11,f11+1):
                    # 	im5_1 = vid.get_data(j)
                    # 	maxProj += np.array(cv2.cvtColor(im5_1,cv2.COLOR_BGR2GRAY))
                    # cEvent.addMaxProj(maxProj)
                    # allEvents.append(cEvent)
                    # maxProj = np.zeros(vDims)
                    if indexes[0] < 5:
                        indexes = np.delete(indexes, 0)
                    if len(indexes) > 1 and indexes[len(indexes)-1] > (f11-s11)-5 and f11 > s11:
                        indexes = np.delete(indexes, len(indexes)-1)
                    if len(indexes) > 0:
                        # print indexes, s11,f11
                        for h in range(0, len(indexes)+1):
                            # print tempEvent.frameStart,tempEvent.frameEnd,s11,f11
                            if h < len(indexes):
                                if indexes[h] >= 5 and indexes[h] < (len(a11)-5):
                                    cEvent = copy.copy(tempEvent)

                                    if h == 0:
                                        cEvent.frameStart = s11
                                        cEvent.frameEnd = indexes[h]+s11-1
                                    elif h > 0 and h < len(indexes):
                                        cEvent.frameStart = indexes[h-1]+s11
                                        cEvent.frameEnd = indexes[h]+s11-1
                                    # print tempEvent.frameStart,tempEvent.frameEnd,s11,f11
                                    for j in range(cEvent.frameStart, cEvent.frameEnd+1):
                                        im5_1 = vid.get_data(j)
                                        maxProj += np.array(cv2.cvtColor(im5_1,
                                                            cv2.COLOR_BGR2GRAY))
                                    cEvent.addMaxProj(maxProj)
                                    # print cEvent.frameStart,cEvent.frameEnd,f11-s11,indexes[h],tempEvent.frameStart,tempEvent.frameEnd
                                    allEvents.append(cEvent)
                                    maxProj = np.zeros(vDims)
                            else:
                                if indexes[h-1] >= 5 and indexes[h-1] < (len(a11)-5):
                                    cEvent = copy.copy(tempEvent)

                                    cEvent.frameStart = indexes[h-1]+s11
                                    cEvent.frameEnd = f11

                                    for j in range(cEvent.frameStart, cEvent.frameEnd+1):
                                        im5_1 = vid.get_data(j)
                                        maxProj += np.array(cv2.cvtColor(im5_1,
                                                            cv2.COLOR_BGR2GRAY))
                                    cEvent.addMaxProj(maxProj)
                                    allEvents.append(cEvent)
                                    maxProj = np.zeros(vDims)
                                else:
                                    if h > 0 and cEvent.frameEnd != f11:
                                        cEvent.frameStart = cEvent.frameEnd
                                        cEvent.frameEnd = f11
                                    else:
                                        cEvent = copy.copy(tempEvent)
                                    for j in range(cEvent.frameStart, cEvent.frameEnd+1):
                                        im5_1 = vid.get_data(j)
                                        maxProj += np.array(cv2.cvtColor(im5_1,
                                                            cv2.COLOR_BGR2GRAY))
                                    cEvent.addMaxProj(maxProj)
                                    allEvents.append(cEvent)
                                    maxProj = np.zeros(vDims)
                    else:
                        for j in range(s11, f11+1):
                            im5_1 = vid.get_data(j)
                            maxProj += np.array(cv2.cvtColor(im5_1,
                                                cv2.COLOR_BGR2GRAY))
                        cEvent.addMaxProj(maxProj)
                        allEvents.append(cEvent)
                        maxProj = np.zeros(vDims)
                    # print f11,num
                    cEvent = simpleBehaviorEvent(num)
                    cEvent.addModeType(cMode)
                    # plt.subplot(121)
                    # plt.plot(a11)
                    # plt.plot(indexes,a11[indexes],"bo")
                    # plt.show()
                else:
                    for j in range(s11, f11+1):
                        im5_1 = vid.get_data(j)
                        maxProj += np.array(cv2.cvtColor(im5_1,
                                            cv2.COLOR_BGR2GRAY))
                    cEvent.addMaxProj(maxProj)
                    allEvents.append(cEvent)
                    maxProj = np.zeros(vDims)
                    cEvent = simpleBehaviorEvent(num)
                    cEvent.addModeType(cMode)
    cEvent.endEvent(num)
    for j in range(s11, f11+1):
        im5_1 = vid.get_data(j)
        maxProj += np.array(cv2.cvtColor(im5_1, cv2.COLOR_BGR2GRAY))
    cEvent.addMaxProj(maxProj)
    allEvents.append(cEvent)
    return allEvents


def eventClassify_withLength(allEvents, COM_x1, COM_y1):
    # for calculating stats about the events produced in eventTabulate.
    # This produces the number of frames spent in each mode type.
    eventType = np.zeros(9)
    for j in range(0, len(allEvents)):
        cM = allEvents[j].getModes()
        [s1, f1] = allEvents[j].getStartAndFinish()
        fCount = f1-s1
        if allEvents[j].quality < 5:
            if 0 in set(cM):
                eventType[0] += fCount  # rest
            elif 1 in set(cM):
                eventType[1] += fCount  # forward, left
            elif 2 in set(cM):
                eventType[2] += fCount  # forward,right
            elif 3 in set(cM):
                eventType[3] += fCount  # back,left
            elif 4 in set(cM):
                eventType[4] += fCount  # back,right
            elif 7 in set(cM):
                eventType[7] += fCount  # stationary left
            elif 8 in set(cM):
                eventType[8] += fCount  # stationary right
            elif 5 in set(cM):
                eventType[5] += fCount  # forward walk
            elif 6 in set(cM):
                eventType[6] += fCount  # backward walk

    return eventType


def eventClassify(allEvents, COM_x1, COM_y1):
    # for calculating stats about the events produced in eventTabulate.
    # This produces the number of occurrences of each mode type.
    eventType = np.zeros(9)
    for j in range(0, len(allEvents)):
        cM = allEvents[j].getModes()
        if allEvents[j].quality < 5:
            if 0 in set(cM):
                eventType[0] += 1  # rest
            elif 1 in set(cM):
                eventType[1] += 1  # forward, left
            elif 2 in set(cM):
                eventType[2] += 1  # forward,right
            elif 3 in set(cM):
                eventType[3] += 1  # back,left
            elif 4 in set(cM):
                eventType[4] += 1  # back,right
            elif 7 in set(cM):
                eventType[7] += 1  # stationary left
            elif 8 in set(cM):
                eventType[8] += 1  # stationary right
            elif 5 in set(cM):
                eventType[5] += 1  # forward walk
            elif 6 in set(cM):
                eventType[6] += 1  # backward walk
    return eventType

>>>>>>> Stashed changes

def plotVelocities():
    indexes = np.unique(np.append(peakutils.indexes(fAngs_deriv, thres=0.2,
                        min_dist=10), peakutils.indexes(-fAngs_deriv, thres=0.2, min_dist=10)))

    print(np.diff(indexes))
    plt.subplot(121)
    plt.plot(fAngs_deriv)
    plt.plot(indexes, fAngs_deriv[indexes], "bo")
    plt.ylabel('angular velocity')
    plt.xlabel('frame')
    # plt.show()

    plt.subplot(122)
    newInds = [ind for ind in np.diff(indexes) if ind > 0]
    plt.hist(newInds)
    plt.xlabel('frames between peaks')
    plt.ylabel('counts')
    plt.show()
    indexes = np.unique(np.append(peakutils.indexes(
        tV, thres=0.2, min_dist=10), peakutils.indexes(-tV, thres=0.2, min_dist=10)))
    #i2 = [abs(fAngs_deriv[indexes[ind]])>0.0  for ind in range(0,len(indexes))]
    #indexes = indexes[i2]
    plt.clf()
    # print indexes
    # print np.diff(indexes)
    plt.subplot(121)
    plt.plot(tV)
    plt.plot(indexes, tV[indexes], "bo")
    plt.ylabel('translational velocity')
    plt.xlabel('frame')

    plt.subplot(122)
    newInds = [ind for ind in np.diff(indexes) if ind > 0]
    plt.hist(newInds)
    plt.xlabel('frames between peaks')
    plt.ylabel('counts')
    plt.show()


def plotWithTurnAngles():
<<<<<<< Updated upstream
	from matplotlib.patches import Wedge
	l = 0
	cT =len(nums)*[(0,0,0,0)]
	vid = imageio.get_reader(filename,  'ffmpeg')#for demoing. 
	inTurn = 0
	img = None
	figScat,ax = plt.subplots()
	if showQuadrants ==1:
		topRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,tH1,90+tV1,alpha =0.1,color = 'orange')
		bottomLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,180+tH1,270+tV1,alpha =0.1,color = 'orange')
		ax.add_patch(topRightQuad)
		ax.add_patch(bottomLeftQuad)
	elif showQuadrants ==2:
		topLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,90.+tV1,180.+tH1,alpha =0.1,color = 'orange')
		bottomRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,270+tV1,360+tH1,alpha =0.1,color = 'orange')
		ax.add_patch(topLeftQuad)
		ax.add_patch(bottomRightQuad)
	scat1 = ax.scatter([],[],s=8)
	for num in nums:
		im5 = vid.get_data(num+nStart)
		if isNearTempBarrier(fc_y[num],fc_x[num],fAngs[num],num)[0]:
			print "near barrier",isNearTempBarrier(fc_y[num],fc_x[num],fAngs[num],num)[1]
		# cTheta = denoisedThetas[num]
		if l<len(allEvents):
			# print allEvents[l].nearTempBarrier
			[s1,f1] = allEvents[l].getStartAndFinish()
			cM = (allEvents[l].getModes()).astype(int)
			#print num,s1,f1,cM,(2 in set(cM) or 8 in set(cM))
			if num ==s1 and (1 in set(cM)):
				print "Moving Left turn"
				cT[num] = (0,0.5,0,0.9)#'g'
				inTurn = 1
			elif num ==s1 and (2 in set(cM)):
				print "Moving Right turn"
				cT[num] = (0,0,1,0.9)#'b'
				inTurn = 1
			elif num ==s1 and (3 in set(cM)):
				print "back left"
				cT[num] = (1,0,1,0.9)#'m'
				inTurn = 1
			elif num ==s1 and (4 in set(cM)):
				print "back right"
				cT[num] = (1,1,0,0.9)#'y'
				inTurn = 1
			elif num ==s1 and (7 in set(cM)):
				print "Stationary Left turn"
				cT[num] = (0,0.5,0,0.9)#'g'
				inTurn = 1
			elif num ==s1 and (8 in set(cM)):
				print "Stationary Right turn"
				cT[num] = (0,0,1,0.9)#'b'
				inTurn = 1
			elif num ==s1 and (5 in set(cM)):
				print "forward"
				cT[num] = (1,0,0,0.9)#'r'
				inTurn = 1
			elif num ==s1 and (6 in set(cM)):
				print "backward"
				cT[num] = (1,0.65,0,0.9)#'orange'
				inTurn = 1
			elif num==s1 and (0 in set(cM)):
				cT[num] = (0,0,0,0.9)#'k'
				inTurn = 1
			elif num >s1 and num <f1 and inTurn==1:
				# for i in xrange(s1,f1+1):
				cT[num] = cT[s1]
				#cT[s1:f1+1] = cT[s1]
			elif num ==f1 and inTurn ==1:
				cT[num] = cT[s1]
				l+=1
				print "Behavior was a total of",fAngs[f1]-fAngs[s1],"degrees.", "Took",f1-s1,"frames.","Distance traveled was",allEvents[l-1].distTraveled
				inTurn=0
			elif num==f1:
				l+=1
		#print s1,f1,cT[s1],num,cT[num]

		imageG5_1 = im5#[25:200,25:200]
		if img is None:
			img = ax.imshow(im5)
		else:
			img.set_data(im5)
		if cT[num]==(0,0,0,0):# or (fc_x[num+1]-fc_x[num])>5:
			fc_x[num] = None
			fc_y[num] = None
		#scat1.set_offsets(np.c_[fc_y[0:num+1]-7*np.sin(np.pi/180.*fAngs[0:num+1]),fc_x[0:num+1]+7*np.cos(np.pi/180*fAngs[0:num+1])])
		scat1.set_offsets(np.c_[fc_y[0:num+1],fc_x[0:num+1]])
		# if np.sqrt((fc_x[num]-q4[1])**2/((halfWidth-7.)**2) + (fc_y[num]-q4[0])**2/((halfHeight-7.)**2)) > 1:
		# 	print "i'm on the wall"
		scat1.set_color(np.c_[cT[0:num+1]])

		plt.pause(0.001)
		if num ==0:
			plt.pause(0.5)
		plt.draw()

def removeShortEvents(listEvents):
	listEvents = filter(lambda x:x.frameStart < x.frameEnd,listEvents)
	return listEvents

def calcDistTraveled(listEvents,center_x,center_y):
	for i1 in xrange(0,len(listEvents)):
		sum1=0.0
		for j1 in xrange((listEvents[i1].frameStart+1),(listEvents[i1].frameEnd+1)):
			sum1 += np.sqrt((center_x[j1]-center_x[j1-1])**2 + (center_y[j1]-center_y[j1-1])**2)
		listEvents[i1].distTraveled = sum1
	return listEvents

def formFullCenterList(center_x,center_y,frames):
	fcenters_x = np.zeros(nFinish-nStart)
	fcenters_y = np.zeros(nFinish-nStart)

	for j in xrange(0,len(center_x)):
		f1 = frames[2*j]
		fcenters_x[f1:f1+len(center_x[j])] = center_x[j]
		fcenters_y[f1:f1+len(center_y[j])] = center_y[j]
	return (fcenters_x,fcenters_y)

def formFullAngleList(dThetas,frames):
	fAngs_1 = np.zeros(nFinish-nStart)
	# print frames,len(dThetas[0]),len(dThetas[1])
	for j1 in xrange(0,len(dThetas)):
		f1_1 = frames[2*j1]
		# print len(fAngs_1[f1_1:f1_1+len(dThetas[j1])]), len(dThetas[j1]),f1_1
		fAngs_1[f1_1:f1_1+len(dThetas[j1])] = dThetas[j1]
	return fAngs_1

def calcQuadrant(x_pos, y_pos):

	#leftRight = y_pos > m1*x_pos + b1
	leftRight =x_pos > y_pos/m1 - b1/m1
	upDown = y_pos > m2*x_pos + b2

	if leftRight and not upDown:
		return 1
	elif leftRight and upDown:
		return 4
	elif not leftRight and upDown:
		return 3
	else:
		return 2
=======
    from matplotlib.patches import Wedge
    l = 0
    cT = len(nums)*[(0, 0, 0, 0)]
    vid = imageio.get_reader(filename,  'ffmpeg')  # for demoing.
    inTurn = 0
    img = None
    figScat, ax = plt.subplots()
    ax.plot(t1[0], t1[1], color='blue')
    ax.plot(t2[0], t2[1], color='green')
    ax.scatter(intersectCenter[0], intersectCenter[1], color='red')
    if showQuadrants == 1:
        topRightQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                             tH1, tV1, alpha=0.1, color='orange')
        bottomLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                               180+tH1, 180+tV1, alpha=0.1, color='orange')
        ax.add_patch(topRightQuad)
        ax.add_patch(bottomLeftQuad)
    elif showQuadrants == 2:
        topLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                            tV1, 180+tH1, alpha=0.1, color='orange')
        bottomRightQuad = Wedge(
            q4, (halfWidth+halfHeight)/2, 180+tV1, 360+tH1, alpha=0.1, color='orange')
        ax.add_patch(topLeftQuad)
        ax.add_patch(bottomRightQuad)
    scat1 = ax.scatter([], [], s=8)
    for num in nums:
        im5 = vid.get_data(num+nStart)
        qHot = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(
            np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
        qHot_rear = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(
            np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
        if isNearTempBarrier(fc_y[num], fc_x[num], fAngs[num], num, qHot, qHot_rear)[0]:
            print("near barrier", isNearTempBarrier(fc_y[num], fc_x[num], fAngs[num], num, qHot, qHot_rear)[2], isQuadHot(
                calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants))
        # 	plt.pause(0.05)
        # 	fig2,ax2 = plt.subplots()
        # 	ax2.scatter(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num]),fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num]),color='orange')
        # 	ax2.plot(t1[0],t1[1],color='blue')
        # 	ax2.plot(t2[0],t2[1],color='green')
        # 	ax2.scatter(intersectCenter[0],intersectCenter[1],color='red')
        # 	plt.pause(20*scaling/4.7)
        # 	plt.close()
        # # cTheta = denoisedThetas[num]
        if l < len(allEvents):
            # print allEvents[l].nearTempBarrier
            [s1, f1] = allEvents[l].getStartAndFinish()
            cM = (allEvents[l].getModes()).astype(int)
            # print num,s1,f1,cM,(2 in set(cM) or 8 in set(cM))
            if num == s1 and (1 in set(cM)):
                print("Moving Left turn")
                cT[num] = (0, 0.5, 0, 0.9)  # 'g'
                inTurn = 1
            elif num == s1 and (2 in set(cM)):
                print("Moving Right turn")
                cT[num] = (0, 0, 1, 0.9)  # 'b'
                inTurn = 1
            elif num == s1 and (3 in set(cM)):
                print("back left")
                cT[num] = (1, 0, 1, 0.9)  # 'm'
                inTurn = 1
            elif num == s1 and (4 in set(cM)):
                print("back right")
                cT[num] = (1, 1, 0, 0.9)  # 'y'
                inTurn = 1
            elif num == s1 and (7 in set(cM)):
                print("Stationary Left turn")
                cT[num] = (0, 0.5, 0, 0.9)  # 'g'
                inTurn = 1
            elif num == s1 and (8 in set(cM)):
                print("Stationary Right turn")
                cT[num] = (0, 0, 1, 0.9)  # 'b'
                inTurn = 1
            elif num == s1 and (5 in set(cM)):
                print("forward")
                cT[num] = (1, 0, 0, 0.9)  # 'r'
                inTurn = 1
            elif num == s1 and (6 in set(cM)):
                print("backward")
                cT[num] = (1, 0.65, 0, 0.9)  # 'orange'
                inTurn = 1
            elif num == s1 and (0 in set(cM)):
                cT[num] = (0, 0, 0, 0.9)  # 'k'
                inTurn = 1
            elif num > s1 and num < f1 and inTurn == 1:
                # for i in range(s1,f1+1):
                cT[num] = cT[s1]
                #cT[s1:f1+1] = cT[s1]
            elif num == f1 and inTurn == 1:
                cT[num] = cT[s1]
                l += 1
                print("Behavior was a total of", fAngs[f1]-fAngs[s1], "degrees.", "Took",
                      f1-s1, "frames.", "Distance traveled was", allEvents[l-1].distTraveled)
                inTurn = 0
            elif num == f1:
                l += 1
        # print s1,f1,cT[s1],num,cT[num]

        imageG5_1 = im5  # [25:20*scaling/4.70,25:20*scaling/4.70]
        if img is None:
            img = ax.imshow(im5)
        else:
            img.set_data(im5)
        if cT[num] == (0, 0, 0, 0):  # or (fc_x[num+1]-fc_x[num])>5:
            fc_x[num] = None
            fc_y[num] = None
        scat1.set_offsets(np.c_[fc_y[0:num+1]-hBL*np.sin(np.pi/180.*fAngs[0:num+1]),
                          fc_x[0:num+1]+hBL*np.cos(np.pi/180*fAngs[0:num+1])])
        # scat1.set_offsets(np.c_[fc_y[0:num+1],fc_x[0:num+1]])
        # if np.sqrt((fc_x[num]-q4[1])**2/((halfWidth-hBL)**2) + (fc_y[num]-q4[0])**2/((halfHeight-hBL)**2)) > 1:
        # 	print "i'm on the wall"
        scat1.set_color(np.c_[cT[0:num+1]])

        plt.pause(0.000000001)
        if num == 0:
            plt.pause(0.00005)
        plt.draw()


def plotWithTurnAngles_turnProbability():
    from matplotlib.patches import Wedge
    l = 0
    cT = len(nums)*[(0, 0, 0, 0)]
    vid = imageio.get_reader(filename,  'ffmpeg')  # for demoing.
    figScat, ax = plt.subplots()
    ax.plot(t1[0], t1[1], color='blue')
    ax.plot(t2[0], t2[1], color='green')
    ax.scatter(intersectCenter[0], intersectCenter[1], color='red')
    inTurn = 0
    img = None
    if showQuadrants == 1:
        topRightQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                             tH1, tV1, alpha=0.1, color='orange')
        bottomLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                               180+tH1, 180+tV1, alpha=0.1, color='orange')
        ax.add_patch(topRightQuad)
        ax.add_patch(bottomLeftQuad)
    elif showQuadrants == 2:
        topLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                            tV1, 180+tH1, alpha=0.1, color='orange')
        bottomRightQuad = Wedge(
            q4, (halfWidth+halfHeight)/2, 180+tV1, 360+tH1, alpha=0.1, color='orange')
        ax.add_patch(topLeftQuad)
        ax.add_patch(bottomRightQuad)
    scat1 = ax.scatter([], [], s=8)
    tAverage = np.zeros(len(nums))
    leftAntDists = np.zeros(len(nums))
    rightAntDists = np.zeros(len(nums))

    for num in nums:
        if num > 0:
            if fc_y[num] == 0 and fc_y[num-1] != 0:
                fc_y[num] = fc_y[num-1]
                fc_x[num] = fc_x[num-1]
                fAngs[num] = fAngs[num-1]

    for num in nums:
        qHot = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(
            np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
        qHot_rear = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(
            np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)

        # get distances of left and right antennae.
        qHotLA = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num])-antennaeDist*np.cos(
            np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])-antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
        qHot_rearLA = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(np.pi/180.*fAngs[num])-antennaeDist*np.cos(
            np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])-antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
        leftAntDists[num] = isNearTempBarrier(fc_y[num]-antennaeDist*np.cos(np.pi/180.*fAngs[num]), fc_x[num] -
                                              antennaeDist*np.sin(np.pi/180.*fAngs[num]), fAngs[num], num, qHot, qHot_rear)[2]*(-1.+2*qHotLA)

        qHotRA = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num]) + antennaeDist*np.cos(
            np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
        qHot_rearRA = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(np.pi/180.*fAngs[num]) + antennaeDist*np.cos(
            np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])+antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
        rightAntDists[num] = isNearTempBarrier(fc_y[num]+antennaeDist*np.cos(np.pi/180.*fAngs[num]), fc_x[num]+antennaeDist*np.sin(
            np.pi/180.*fAngs[num]), fAngs[num], num, qHot, qHot_rear)[2]*(-1.+2*qHotRA)

    (yvals40, levels40, yvals30, levels30, yvals35, levels35, x2, y2, ti, ti0,
     ti35) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")
    condition = filename.split('_')[1]
    if condition == '25vs40':
        col1 = 'red'
        yvals = yvals40
        levels = levels40
        ti1 = ti0
    elif condition == '25vs30':
        col1 = 'yellow'
        yvals = yvals30
        levels = levels30
        ti1 = ti
    elif condition == '25vs35':
        col1 = 'orange'
        yvals = yvals35
        levels = levels35
        ti1 = ti35
    # calculate temperatures at each antenna
    indsColdLA = leftAntDists < -12*scaling/4.7
    indsHotLA = leftAntDists > 20*scaling/4.7
    indsMiddleLA = [leftAntDists[i] > -12*scaling/4.7 and leftAntDists[i]
                    < 20*scaling/4.7 for i in range(0, len(leftAntDists))]

    minT = np.min(ti1[:, 1])
    maxT = np.max(ti1[:, 1])
    tempsLA = np.zeros(len(nums))
    tempsRA = np.zeros(len(nums))
    tempsLA[indsColdLA] = minT
    tempsLA[indsHotLA] = maxT
    from scipy.interpolate import griddata
    tVals = griddata(y2[:, 1]*scaling, ti1[:, 1], rightAntDists[indsMiddleLA])
    tempsLA[indsMiddleLA] = tVals

    indsColdRA = rightAntDists < -12*scaling/4.7
    indsHotRA = rightAntDists > 20*scaling/4.7
    indsMiddleRA = [rightAntDists[i] > -12*scaling/4.7 and rightAntDists[i]
                    < 20*scaling/4.7 for i in range(0, len(rightAntDists))]

    tempsRA[indsColdRA] = minT
    tempsRA[indsHotRA] = maxT
    import matplotlib
    import matplotlib.cm as cm

    tVals = griddata(y2[:, 1]*scaling, ti1[:, 1], leftAntDists[indsMiddleRA])
    tempsRA[indsMiddleRA] = tVals
    mTL = np.max(tempsLA)
    minTL = np.min(tempsLA)
    # read in filters and scaling info
    (filtT, filtTd, minT, minT2, scaleT, scaleT2, case) = pickle.load(
        open("filter_data.pkl", "rb"), encoding="bytes")
    # calculate
    if case == 1:
        tAverage = np.concatenate(
            ((np.array(30*[0]), ((tempsLA+tempsRA)/2)-minT)/scaleT))
        tDifference = np.concatenate(
            ((np.array(30*[0])), ((tempsRA-tempsLA)-minT2)/scaleT2))
    elif case == 2:
        tAverage = np.concatenate(
            (np.array(30*[0]), ((np.gradient(tempsLA+tempsRA)/2)-minT)/scaleT))
        tDifference = np.concatenate(
            ((np.array(30*[0])), (np.gradient(np.abs(tempsRA-tempsLA))-minT2)/scaleT2))

    currentT = np.zeros(len(nums))
    currentTd = np.zeros(len(nums))
    for num in nums:
        currentT[num] = np.dot(filtT, tAverage[num+1:num+31])
        # plt.clf()
        # plt.plot(tAverage[30:num+31],color='black')
        # plt.plot(currentT[0:num+1])
        # print currentT[0:num]
        # plt.show()
        currentTd[num] = np.dot(filtTd, tDifference[num+1:num+31])

    fullSum = currentT+currentTd
    fullSum = fullSum  # - np.min(fullSum)
    # plt.clf()
    # # plt.plot(fullSum,color='red')
    # plt.plot(np.gradient(tAverage),color='black')
    # plt.plot(range(30,len(tAverage)),currentT)
    # plt.show()
    norm = matplotlib.colors.Normalize(vmin=np.min(
        fullSum)*0.25, vmax=np.max(fullSum)*0.25, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hot)
    fullSum2 = []
    for i in range(0, len(fullSum)):
        fullSum2.append(mapper.to_rgba(fullSum[i]))

    for num in nums:
        im5 = vid.get_data(num+nStart)

        # print s1,f1,cT[s1],num,cT[num]
        imageG5_1 = im5  # [25:20*scaling/4.70,25:20*scaling/4.70]
        if img is None:
            img = ax.imshow(im5)
        else:
            img.set_data(im5)
        # if cT[num]==(0,0,0,0):# or (fc_x[num+1]-fc_x[num])>5:
        # 	fc_x[num] = None
        # 	fc_y[num] = None
        scat1.set_offsets(np.c_[fc_y[np.max([0, num-150]):num+1]-hBL*np.sin(np.pi/180.*fAngs[np.max([0, num-150]):num+1]),
                          fc_x[np.max([0, num-150]):num+1]+hBL*np.cos(np.pi/180*fAngs[np.max([0, num-150]):num+1])])
        cT[num] = fullSum2[num]  # (fullSum[num],0,0,0.8)
        # scat1.set_offsets(np.c_[fc_y[0:num+1],fc_x[0:num+1]])
        # if np.sqrt((fc_x[num]-q4[1])**2/((halfWidth-hBL)**2) + (fc_y[num]-q4[0])**2/((halfHeight-hBL)**2)) > 1:
        # 	print "i'm on the wall"
        scat1.set_color(np.c_[cT[np.max([0, num-150]):num+1]])
        # print tempsLA[num],tempsRA[num]
        plt.pause(0.0001)
        if num == 0:
            plt.pause(0.5)
        plt.draw()


def plotWithTurnAngles_turnProbability_vidsave():
    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Fly_track', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=30, metadata=metadata)
    from matplotlib.patches import Wedge
    figScat, ax = plt.subplots()
    with writer.saving(figScat, filename.split(".")[0].split("/")[-1]+"_track_turnProb.mp4", 100):
        from matplotlib.patches import Wedge
        l = 0
        cT = len(nums)*[(0, 0, 0, 0)]
        vid = imageio.get_reader(filename,  'ffmpeg')  # for demoing.
        inTurn = 0
        img = None
        if showQuadrants == 1:
            topRightQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                                 tH1, tV1, alpha=0.1, color='orange')
            bottomLeftQuad = Wedge(
                q4, (halfWidth+halfHeight)/2, 180+tH1, 180+tV1, alpha=0.1, color='orange')
            ax.add_patch(topRightQuad)
            ax.add_patch(bottomLeftQuad)
        elif showQuadrants == 2:
            topLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                                tV1, 180+tH1, alpha=0.1, color='orange')
            bottomRightQuad = Wedge(
                q4, (halfWidth+halfHeight)/2, 180+tV1, 360+tH1, alpha=0.1, color='orange')
            ax.add_patch(topLeftQuad)
            ax.add_patch(bottomRightQuad)
        scat1 = ax.scatter([], [], s=8)
        tAverage = np.zeros(len(nums))
        leftAntDists = np.zeros(len(nums))
        rightAntDists = np.zeros(len(nums))

        for num in nums:
            if num > 0:
                if fc_y[num] == 0 and fc_y[num-1] != 0:
                    fc_y[num] = fc_y[num-1]
                    fc_x[num] = fc_x[num-1]
                    fAngs[num] = fAngs[num-1]

        for num in nums:
            qHot = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(
                np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
            qHot_rear = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(
                np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)

            # get distances of left and right antennae.
            qHotLA = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num])-antennaeDist*np.cos(
                np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])-antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
            qHot_rearLA = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(np.pi/180.*fAngs[num])-antennaeDist*np.cos(
                np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])-antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
            leftAntDists[num] = isNearTempBarrier(fc_y[num]-antennaeDist*np.cos(np.pi/180.*fAngs[num]), fc_x[num]-antennaeDist*np.sin(
                np.pi/180.*fAngs[num]), fAngs[num], num, qHot, qHot_rear)[2]*(-1.+2*qHotLA)

            qHotRA = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num]) + antennaeDist*np.cos(
                np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
            qHot_rearRA = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(np.pi/180.*fAngs[num]) + antennaeDist*np.cos(
                np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])+antennaeDist*np.sin(np.pi/180.*fAngs[num])), showQuadrants)
            rightAntDists[num] = isNearTempBarrier(fc_y[num]+antennaeDist*np.cos(np.pi/180.*fAngs[num]), fc_x[num]+antennaeDist*np.sin(
                np.pi/180.*fAngs[num]), fAngs[num], num, qHot, qHot_rear)[2]*(-1.+2*qHotRA)

        (yvals40, levels40, yvals30, levels30, yvals35, levels35, x2, y2, ti, ti0,
         ti35) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")
        condition = filename.split('_')[1]
        if condition == '25vs40':
            col1 = 'red'
            yvals = yvals40
            levels = levels40
            ti1 = ti0
        elif condition == '25vs30':
            col1 = 'yellow'
            yvals = yvals30
            levels = levels30
            ti1 = ti
        elif condition == '25vs35':
            col1 = 'orange'
            yvals = yvals35
            levels = levels35
            ti1 = ti35
        # calculate temperatures at each antenna
        indsColdLA = leftAntDists < -12*scaling/4.7
        indsHotLA = leftAntDists > 20*scaling/4.7
        indsMiddleLA = [leftAntDists[i] > -12*scaling/4.7 and leftAntDists[i]
                        < 20*scaling/4.7 for i in range(0, len(leftAntDists))]

        minT = np.min(ti1[:, 1])
        maxT = np.max(ti1[:, 1])
        tempsLA = np.zeros(len(nums))
        tempsRA = np.zeros(len(nums))
        tempsLA[indsColdLA] = minT
        tempsLA[indsHotLA] = maxT
        from scipy.interpolate import griddata
        tVals = griddata(y2[:, 1]*scaling, ti1[:, 1],
                         rightAntDists[indsMiddleLA])
        tempsLA[indsMiddleLA] = tVals

        indsColdRA = rightAntDists < -12*scaling/4.7
        indsHotRA = rightAntDists > 20*scaling/4.7
        indsMiddleRA = [rightAntDists[i] > -12*scaling/4.7 and rightAntDists[i]
                        < 20*scaling/4.7 for i in range(0, len(rightAntDists))]

        tempsRA[indsColdRA] = minT
        tempsRA[indsHotRA] = maxT
        import matplotlib
        import matplotlib.cm as cm

        tVals = griddata(y2[:, 1]*scaling, ti1[:, 1],
                         leftAntDists[indsMiddleRA])
        tempsRA[indsMiddleRA] = tVals
        mTL = np.max(tempsLA)
        minTL = np.min(tempsLA)
        # read in filters and scaling info
        (filtT, filtTd, minT, minT2, scaleT, scaleT2, case) = pickle.load(
            open("filter_data.pkl", "rb"), encoding="bytes")
        # calculate
        if case == 1:
            tAverage = np.concatenate(
                ((np.array(30*[0]), ((tempsLA+tempsRA)/2)-minT)/scaleT))
            tDifference = np.concatenate(
                ((np.array(30*[0])), ((tempsRA-tempsLA)-minT2)/scaleT2))
        elif case == 2:
            tAverage = np.concatenate(
                (np.array(30*[0]), ((np.gradient(tempsLA+tempsRA)/2)-minT)/scaleT))
            tDifference = np.concatenate(
                ((np.array(30*[0])), (np.gradient(np.abs(tempsRA-tempsLA))-minT2)/scaleT2))

        currentT = np.zeros(len(nums))
        currentTd = np.zeros(len(nums))
        for num in nums:
            currentT[num] = np.dot(filtT, tAverage[num+1:num+31])
            currentTd[num] = np.dot(filtTd, tDifference[num+1:num+31])

        fullSum = currentT+currentTd
        fullSum = fullSum - np.min(fullSum)
        norm = matplotlib.colors.Normalize(
            vmin=0., vmax=np.max(fullSum), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.hot)
        fullSum2 = []
        for i in range(0, len(fullSum)):
            fullSum2.append(mapper.to_rgba(fullSum[i]))

        for num in nums:
            im5 = vid.get_data(num+nStart)

            # print s1,f1,cT[s1],num,cT[num]

            imageG5_1 = im5  # [25:20*scaling/4.70,25:20*scaling/4.70]
            if img is None:
                img = ax.imshow(im5)
            else:
                img.set_data(im5)
            # if cT[num]==(0,0,0,0):# or (fc_x[num+1]-fc_x[num])>5:
            # 	fc_x[num] = None
            # 	fc_y[num] = None
            scat1.set_offsets(np.c_[fc_y[np.max([0, num-150]):num+1]-hBL*np.sin(np.pi/180.*fAngs[np.max([0, num-150]):num+1]),
                              fc_x[np.max([0, num-150]):num+1]+hBL*np.cos(np.pi/180*fAngs[np.max([0, num-150]):num+1])])
            cT[num] = fullSum2[num]  # (fullSum[num],0,0,0.8)
            # scat1.set_offsets(np.c_[fc_y[0:num+1],fc_x[0:num+1]])
            # if np.sqrt((fc_x[num]-q4[1])**2/((halfWidth-hBL)**2) + (fc_y[num]-q4[0])**2/((halfHeight-hBL)**2)) > 1:
            # 	print "i'm on the wall"
            scat1.set_color(np.c_[cT[np.max([0, num-150]):num+1]])
            writer.grab_frame()
            # print tempsLA[num],tempsRA[num]
            # plt.pause(0.0001)
            # if num ==0:
            # 	plt.pause(0.5)
            # plt.draw()


def plotWithTurnAngles_vidsave():
    # modified for supp figure... fix later.
    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Fly_track', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=30*8, metadata=metadata)
    from matplotlib.patches import Wedge
    figScat, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    with writer.saving(figScat, filename.split(".")[0].split("/")[-1]+"_track_color.mp4", 100):
        l = 0
        cT = len(nums)*[(0, 0, 0, 0)]
        vid = imageio.get_reader(filename,  'ffmpeg')  # for demoing.
        inTurn = 0
        img = None
        # ax.plot(t1[0],t1[1],color='blue')
        # ax.plot(t2[0],t2[1],color='green')
        # ax.scatter(intersectCenter[0],intersectCenter[1],color='red')
        if showQuadrants == 1:
            topRightQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                                 tH1, tV1, alpha=0.3, color='pink')
            bottomLeftQuad = Wedge(
                q4, (halfWidth+halfHeight)/2, 180+tH1, 180+tV1, alpha=0.3, color='pink')
            lq = ax.add_patch(topRightQuad)
            rq = ax.add_patch(bottomLeftQuad)
        elif showQuadrants == 2:
            topLeftQuad = Wedge(q4, (halfWidth+halfHeight)/2,
                                tV1, 180+tH1, alpha=0.3, color='pink')
            bottomRightQuad = Wedge(
                q4, (halfWidth+halfHeight)/2, 180+tV1, 360+tH1, alpha=0.3, color='pink')
            lq = ax.add_patch(topLeftQuad)
            rq = ax.add_patch(bottomRightQuad)
        # scat1 = ax.scatter([],[],s=8)
        # alphaL = np.linspace(0,0.3,300)
        for num in nums:
            # if num < 300:
            # 	lq.set_alpha(alphaL[num])
            # 	rq.set_alpha(alphaL[num])
            im5 = vid.get_data(num+nStart)
            qHot = isQuadHot(calcQuadrant(fc_y[num]-hBL*np.sin(
                np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
            qHot_rear = isQuadHot(calcQuadrant(fc_y[num]+hBL*np.sin(
                np.pi/180.*fAngs[num]), fc_x[num]-hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants)
            if isNearTempBarrier(fc_y[num], fc_x[num], fAngs[num], num, qHot, qHot_rear)[0]:
                print("near barrier", isNearTempBarrier(fc_y[num], fc_x[num], fAngs[num], num, qHot, qHot_rear)[2], isQuadHot(
                    calcQuadrant(fc_y[num]-hBL*np.sin(np.pi/180.*fAngs[num]), fc_x[num]+hBL*np.cos(np.pi/180*fAngs[num])), showQuadrants))
                # plt.pause(0.05)
            # cTheta = denoisedThetas[num]
            if l < len(allEvents):
                # print allEvents[l].nearTempBarrier
                [s1, f1] = allEvents[l].getStartAndFinish()
                cM = (allEvents[l].getModes()).astype(int)
                # print num,s1,f1,cM,(2 in set(cM) or 8 in set(cM))
                if num == s1 and (1 in set(cM)):
                    print("Moving Left turn")
                    cT[num] = (0, 0.5, 0, 0.9)  # 'g'
                    inTurn = 1
                elif num == s1 and (2 in set(cM)):
                    print("Moving Right turn")
                    cT[num] = (0, 0, 1, 0.9)  # 'b'
                    inTurn = 1
                elif num == s1 and (3 in set(cM)):
                    print("back left")
                    cT[num] = (1, 0, 1, 0.9)  # 'm'
                    inTurn = 1
                elif num == s1 and (4 in set(cM)):
                    print("back right")
                    cT[num] = (1, 1, 0, 0.9)  # 'y'
                    inTurn = 1
                elif num == s1 and (7 in set(cM)):
                    print("Stationary Left turn")
                    cT[num] = (0, 0.5, 0, 0.9)  # 'g'
                    inTurn = 1
                elif num == s1 and (8 in set(cM)):
                    print("Stationary Right turn")
                    cT[num] = (0, 0, 1, 0.9)  # 'b'
                    inTurn = 1
                elif num == s1 and (5 in set(cM)):
                    print("forward")
                    cT[num] = (1, 0, 0, 0.9)  # 'r'
                    inTurn = 1
                elif num == s1 and (6 in set(cM)):
                    print("backward")
                    cT[num] = (1, 0.65, 0, 0.9)  # 'orange'
                    inTurn = 1
                elif num == s1 and (0 in set(cM)):
                    cT[num] = (0, 0, 0, 0.9)  # 'k'
                    inTurn = 1
                elif num > s1 and num < f1 and inTurn == 1:
                    # for i in range(s1,f1+1):
                    cT[num] = cT[s1]
                    #cT[s1:f1+1] = cT[s1]
                elif num == f1 and inTurn == 1:
                    cT[num] = cT[s1]
                    l += 1
                    print("Behavior was a total of", fAngs[f1]-fAngs[s1], "degrees.", "Took",
                          f1-s1, "frames.", "Distance traveled was", allEvents[l-1].distTraveled)
                    inTurn = 0
                elif num == f1:
                    l += 1
            # print s1,f1,cT[s1],num,cT[num]

            imageG5_1 = im5  # [25:20*scaling/4.70,25:20*scaling/4.70]
            if img is None:
                img = ax.imshow(im5)
            else:
                img.set_data(im5)
            if cT[num] == (0, 0, 0, 0):  # or (fc_x[num+1]-fc_x[num])>5:
                fc_x[num] = None
                fc_y[num] = None
            # scat1.set_offsets(np.c_[fc_y[0:num+1]-hBL*np.sin(np.pi/180.*fAngs[0:num+1]),fc_x[0:num+1]+hBL*np.cos(np.pi/180*fAngs[0:num+1])])
            # # scat1.set_offsets(np.c_[fc_y[0:num+1],fc_x[0:num+1]])
            # # if np.sqrt((fc_x[num]-q4[1])**2/((halfWidth-hBL)**2) + (fc_y[num]-q4[0])**2/((halfHeight-hBL)**2)) > 1:
            # # 	print "i'm on the wall"
            # scat1.set_color(np.c_[cT[0:num+1]])
            writer.grab_frame()
            # plt.pause(0.0001)
            # if num ==0:
            # 	plt.pause(0.5)
            # plt.draw()


def removeShortEvents(listEvents):
    listEvents = filter(lambda x: x.frameStart < x.frameEnd, listEvents)
    return list(listEvents)  # filter not iterable in python3


def calcDistTraveled(listEvents, center_x, center_y):
    for i1 in range(0, len(listEvents)):
        sum1 = 0.0
        for j1 in range((listEvents[i1].frameStart+1), (listEvents[i1].frameEnd+1)):
            sum1 += np.sqrt((center_x[j1]-center_x[j1-1])
                            ** 2 + (center_y[j1]-center_y[j1-1])**2)
        listEvents[i1].distTraveled = sum1
    return listEvents


def formFullCenterList(center_x, center_y, frames):
    fcenters_x = np.zeros(nFinish-nStart)
    fcenters_y = np.zeros(nFinish-nStart)

    for j in range(0, len(center_x)):
        f1 = frames[2*j]
        if (frames[2*j+1]-frames[2*j]) >= 10:
            fcenters_x[f1:f1+len(center_x[j])] = center_x[j]
            fcenters_y[f1:f1+len(center_y[j])] = center_y[j]
    return (fcenters_x, fcenters_y)


def formFullAngleList(dThetas, frames):
    fAngs_1 = np.zeros(nFinish-nStart)
    # print frames,len(dThetas[0]),len(dThetas[1])
    for j1 in range(0, len(dThetas)):
        f1_1 = frames[2*j1]
        if (frames[2*j1+1]-frames[2*j1]) >= 10:
            # print len(fAngs_1[f1_1:f1_1+len(dThetas[j1])]), len(dThetas[j1]),f1_1
            fAngs_1[f1_1:f1_1+len(dThetas[j1])] = dThetas[j1]
    return fAngs_1


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

>>>>>>> Stashed changes

def quadrantTimeSpent(quad_full_1):

    one = np.count_nonzero(quad_full_1 == 1.0)
    two = np.count_nonzero(quad_full_1 == 2.0)
    three = np.count_nonzero(quad_full_1 == 3.0)
    four = np.count_nonzero(quad_full_1 == 4.0)
    return (one+three, two+four)


def quadrantFillInMissing(quad_full_1):
<<<<<<< Updated upstream
	lastQuad = None
	fillIn = 0
	for i1 in xrange(0,len(quad_full_1)):
		if quad_full_1[i1] == 0 and lastQuad == None:
			fillIn +=1
		else:
			if quad_full_1[i1] ==0:
				quad_full_1[i1] = lastQuad
			else:
				lastQuad = quad_full_1[i1]

	quad_full_1[0:fillIn] = quad_full_1[fillIn]  
	return quad_full_1   

def isNearTempBarrier(x_pos,y_pos,theta,num):
	cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
	cVector = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])+7.0*cToHead
	hVec = cVector - np.dot(cVector,horizVector)*horizVector
	vVec = cVector - np.dot(cVector,vertVector)*vertVector
	hDist = np.linalg.norm(hVec)
	vDist = np.linalg.norm(vVec)
	# print hDist,vDist
	#a = cVector+7.0*cToHead
	# to check geometry
	# print hDist,vDist
	#vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.

	mDist = np.min([hDist,vDist])
	if hDist <nearThresh and hDist<vDist:
		cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
		hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector

		tVal = np.arccos(np.dot(cToHead,horizVector))*180./np.pi
		if np.linalg.norm(hVec_rear)<hDist:# np.dot(hVec,hVec_rear)<0 or 

			return 0,0,mDist
		if hVec_rear[1]<0:
			tVal = 180.-tVal
		# print tVal,np.arccos(np.dot(cToHead,vertVector))*180./np.pi
		return 1,tVal,mDist
	elif vDist <nearThresh and vDist<=hDist:
		cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
		vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
		
		# im5 = vid.get_data(num+nStart)
		# cRear1 = cRear+ np.array(intersectCenter)
		# chead1 = cVector+ np.array(intersectCenter)
		# plt.imshow(im5)
		# plt.plot(cRear1[0],cRear1[1],'ro')
		# plt.plot(chead1[0],chead1[1],'go')
		# plt.show()
		# plt.plot([0,50*horizVector[0]],[0,50*horizVector[1]],color='k',lw=5)
		# plt.plot([0,50*vertVector[0]],[0,50*vertVector[1]],color = 'g',lw=5)
		# plt.plot(cVector[0],cVector[1],'ko')
		# a11 = np.array([x_pos-intersectCenter[0],y_pos-intersectCenter[1]])
		# plt.plot(a11[0],a11[1],'go')
		# b11 = a11 - 7.0*cToHead 
		# plt.plot(b11[0],b11[1],'ro')
		# #plt.plot(a[0]+intersectCenter[0],intersectCenter[1]+a[1],'o')
		# plt.plot(t1[0]-intersectCenter[0],t1[1]-intersectCenter[1],color='r')
		# plt.plot(t2[0]-intersectCenter[0],t2[1]-intersectCenter[1],color='cyan')
		# # cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
		# # hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector
		# # cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
		# # vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
		# # print np.dot(hVec,hVec_rear),np.linalg.norm(hVec_rear),hDist
		# # print np.dot(vVec,vVec_rear),np.linalg.norm(vVec_rear),vDist
		# plt.show()
		tVal = np.arccos(np.dot(cToHead,vertVector))*180./np.pi
		if np.linalg.norm(vVec_rear)<vDist:#np.dot(vVec,vVec_rear)<0 or
			return 0,0,mDist
		if vVec_rear[0]>0:
			tVal = 180.-tVal
		return 1,tVal,mDist
	else:
		return 0,0,mDist
		
# def isNearTempBarrier(x_pos,y_pos,theta,num):
# 	cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])#np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
# 	cVector = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])+7.0*cToHead
# 	hVec = cVector - np.dot(cVector,horizVector)*horizVector
# 	vVec = cVector - np.dot(cVector,vertVector)*vertVector
# 	hDist = np.linalg.norm(hVec)
# 	vDist = np.linalg.norm(vVec)
# 	# print hDist,vDist
# 	#a = cVector+7.0*cToHead
# 	# to check geometry
# 	# print hDist,vDist
# 	#vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.

# 	if hDist <nearThresh and hDist<vDist:
# 		cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
# 		hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector
# 		if np.linalg.norm(hVec_rear)<hDist:# np.dot(hVec,hVec_rear)<0 or 

# 			return 0,0

# 		tVal = np.arccos(np.dot(cToHead,horizVector))*180./np.pi
# 		if hVec_rear[1]<0:
# 			tVal = 180.-tVal
# 		# print tVal,np.arccos(np.dot(cToHead,vertVector))*180./np.pi
# 		return 1,tVal
# 	elif vDist <nearThresh and vDist<=hDist:
# 		cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
# 		vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
		
# 		# im5 = vid.get_data(num+nStart)
# 		# cRear1 = cRear+ np.array(intersectCenter)
# 		# chead1 = cVector+ np.array(intersectCenter)
# 		# plt.imshow(im5)
# 		# plt.plot(cRear1[0],cRear1[1],'ro')
# 		# plt.plot(chead1[0],chead1[1],'go')
# 		# plt.show()
# 		# plt.plot([0,50*horizVector[0]],[0,50*horizVector[1]],color='k',lw=5)
# 		# plt.plot([0,50*vertVector[0]],[0,50*vertVector[1]],color = 'g',lw=5)
# 		# plt.plot(cVector[0],cVector[1],'ko')
# 		# a11 = np.array([x_pos-intersectCenter[0],y_pos-intersectCenter[1]])
# 		# plt.plot(a11[0],a11[1],'go')
# 		# b11 = a11 - 7.0*cToHead 
# 		# plt.plot(b11[0],b11[1],'ro')
# 		# #plt.plot(a[0]+intersectCenter[0],intersectCenter[1]+a[1],'o')
# 		# plt.plot(t1[0]-intersectCenter[0],t1[1]-intersectCenter[1],color='r')
# 		# plt.plot(t2[0]-intersectCenter[0],t2[1]-intersectCenter[1],color='cyan')
# 		# cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
# 		# hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector
# 		# cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-7.0*cToHead
# 		# vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
# 		# print np.dot(hVec,hVec_rear),np.linalg.norm(hVec_rear),hDist
# 		# print np.dot(vVec,vVec_rear),np.linalg.norm(vVec_rear),vDist
# 		# plt.show()

# 		if np.linalg.norm(vVec_rear)<vDist:#np.dot(vVec,vVec_rear)<0 or

# 			return 0,0
# 		tVal = np.arccos(np.dot(cToHead,vertVector))*180./np.pi
# 		if vVec_rear[0]>0:
# 			tVal = 180.-tVal
# 		return 1,tVal
# 	else:
# 		return 0,0
def fullQuadCalc(x_list,y_list):
	quadNum = np.zeros(len(x_list))
	for i in xrange(0,len(x_list)):
		quadNum[i] = calcQuadrant(y_list[i],x_list[i])

	return quadNum

if __name__ == "__main__":
	import imageio
	import cv2
	import os
	import matplotlib.pyplot as plt
	from matplotlib import animation
	import numpy as np
	import sys
	import pickle
	import StringIO
	from PIL import Image

	centerType = "COM"

	saveData = True
	#filenames = os.listdir(vidDir)
	#print sys.argv
	filename = str(sys.argv[1])
	showQuadrants = int(filename.split("/")[-1].split(".")[0].split("_")[-1])
	toPlot = int(sys.argv[2])
	arenaFile = str(sys.argv[3])
	outputFolder = str(sys.argv[4])
	#filename = vidDir + "/" +fname #filenames[1]
	
	data = pickle.load(open(arenaFile,"rb"))
	q4 = list(data[0])

	halfWidth = data[1]/2.
	halfHeight = data[2]/2.
	t1 = data[3]
	t2 = data[4]

	m1 = (t1[1][1]-t1[1][0])/(t1[0][1]-t1[0][0])

	m2 = (t2[1][1]-t2[1][0])/(t2[0][1]-t2[0][0])

	b1 = -m1*t1[0][1] + t1[1][1]
	b2 = -m2*t2[0][1] + t2[1][1]

	intersectCenter = np.array([(b2-b1)/(m1-m2),m1*(b2-b1)/(m1-m2) + b1])
	vertVector = np.array([(t1[0][1]-t1[0][0]),(t1[1][1]-t1[1][0])])
	vertVector = vertVector/np.linalg.norm(vertVector)

	if vertVector[1]<0:
		vertVector = -vertVector

	horizVector = np.array([(t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0])])
	horizVector = horizVector/np.linalg.norm(horizVector)
	if horizVector[0]<0:
		horizVector = -horizVector
	nearThresh = 10

	tH1 = 180./np.pi*np.arctan2(horizVector[1],horizVector[0])
	tV1 =180./np.pi*np.arctan2(vertVector[0],vertVector[1])

	#extract positional/rotational information, plus associated velocities. 
	(COM_x1,COM_y1, vx_1,vy_1,denoisedThetas,denoisedThetas_deriv,nStart,nFinish,al1s,al2s,frames,quad,qH,qB) = coreDataExtract(filename)
#     #now decompose velocity along the direction of the fly's major axis. 
	(transV,slipV) = decomposeVelocity(vx_1,vy_1,denoisedThetas)
	(redo,denoisedThetas) = checkOrientation(transV,denoisedThetas)
	if redo ==1:
		print redo
		(transV,slipV) = decomposeVelocity(vx_1,vy_1,denoisedThetas)
	nums = xrange(0,nFinish-nStart)
	#if most translation is backward, rotate theta by 180 degress and repeat velocity decomp. 


	(mode,modeQuality,onWall,tempB,tempB_dist) = timepointClassify(transV,denoisedThetas_deriv,al1s,al2s,COM_x1,COM_y1,frames)

	(fc_x,fc_y) = formFullCenterList(COM_x1,COM_y1,frames)
	fAngs = formFullAngleList(denoisedThetas,frames)
	fAngs_deriv = formFullAngleList(denoisedThetas_deriv,frames)
	tV = formFullAngleList(transV,frames)
	sV = formFullAngleList(slipV,frames)

	head_x = fc_x+7*np.cos(np.pi/180*fAngs)
	head_y = fc_y-7*np.sin(np.pi/180.*fAngs)
	back_x = fc_x-7*np.cos(np.pi/180*fAngs)
	back_y = fc_y+7*np.sin(np.pi/180.*fAngs)
	quad_full = fullQuadCalc(fc_x,fc_y)
	quad_head_full = fullQuadCalc(head_x,head_y)
	quad_back_full = fullQuadCalc(back_x,back_y)
	# quad_full = formFullAngleList(quad,frames)
	# quad_full = quadrantFillInMissing(quad_full)
	cold,hot = quadrantTimeSpent(quad_full)
	# print len(quad_head_full)
	# quad_head_full = formFullAngleList(qH,frames)
	# quad_head_full = quadrantFillInMissing(quad_head_full)

	# quad_back_full = formFullAngleList(qB,frames)
	# quad_back_full = quadrantFillInMissing(quad_back_full)

	if showQuadrants ==2:
		swap1 = hot
		hot = cold
		cold = swap1


	# tV2 = [np.sqrt(tV[j]**2 + sV[j]**2) for j in xrange(0,len(sV))]
	allEvents = eventTabulate(mode,nStart,nFinish,onWall,quad_full,tempB,tempB_dist,quad_head_full,quad_back_full)
	allEvents = removeShortEvents(allEvents)
	allEvents = calcDistTraveled(allEvents,fc_x,fc_y)

	eventType = eventClassify(allEvents,fc_x,fc_y)
	eventType2 = eventClassify_withLength(allEvents,fc_x,fc_y)

	if toPlot ==1:
		plotWithTurnAngles()
	elif toPlot ==2:
		plotVelocities()

	if saveData ==True:
		trajData = (allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs)
		outFile = outputFolder + '/'+ filename.split(".")[0].split("/")[-1] + ".output"
		pickle.dump(trajData,open(outFile,"wb"))

	print "===Results==="
	print "Frames of resting:",eventType2[0]
	print "Frames of forward moving left turns:",eventType2[1]
	print "Frames of forward moving right turns:",eventType2[2]
	print "Frames of backward moving left turns:",eventType2[3]
	print "Frames of backward moving right turns:",eventType2[4]
	print "Frames of pure forward moves:",eventType2[5]
	print "Frames of pure backward moves:",eventType2[6]
	print "Frames of stationary left turns:",eventType2[7]
	print "Frames of stationary right turns:",eventType2[8]
	print "Frames in hot regions:", hot
	print "Frames in cold regions",cold
=======
    lastQuad = None
    fillIn = 0
    for i1 in range(0, len(quad_full_1)):
        if quad_full_1[i1] == 0 and lastQuad == None:
            fillIn += 1
        else:
            if quad_full_1[i1] == 0:
                quad_full_1[i1] = lastQuad
            else:
                lastQuad = quad_full_1[i1]

    quad_full_1[0:fillIn] = quad_full_1[fillIn]
    return quad_full_1


def isNearTempBarrier(x_pos, y_pos, theta, num, quad_hot, quad_hot_rear):
    # np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
    cToHead = np.array([-np.sin(np.pi/180.*theta), np.cos(np.pi/180.*theta)])
    cVector = np.array([x_pos - intersectCenter[0],
                       y_pos - intersectCenter[1]])+hBL*cToHead
    hVec = cVector - np.dot(cVector, horizVector)*horizVector
    vVec = cVector - np.dot(cVector, vertVector)*vertVector
    hDist = np.linalg.norm(hVec)
    vDist = np.linalg.norm(vVec)

    mDist = np.min([hDist, vDist])
    if mDist < nearThresh_temp:

        if hDist < vDist:
            cRear = np.array([x_pos - intersectCenter[0],
                             y_pos - intersectCenter[1]])-hBL*cToHead
            hVec_rear = cRear - np.dot(cRear, horizVector)*horizVector
            hdistRear = np.linalg.norm(hVec_rear)
            tVal = np.arccos(np.dot(cToHead, horizVector))*180./np.pi

            if quad_hot:
                hDist *= -1.

            if quad_hot_rear:
                hdistRear *= -1.

            # if hVec_rear[1]<0 or (hdistRear<0 and hVec_rear[1]>0):
            # 	tVal = 180.-tVal

            if hDist > hLimit or hDist < lLimit or hdistRear < hDist:
                return 0, 0, hDist

            return 1, tVal, hDist

        elif vDist <= hDist:
            cRear = np.array([x_pos - intersectCenter[0],
                             y_pos - intersectCenter[1]])-hBL*cToHead
            vVec_rear = cRear - np.dot(cRear, vertVector)*vertVector
            vDistRear = np.linalg.norm(vVec_rear)

            # im5 = vid.get_data(num+nStart)
            # cRear1 = cRear+ np.array(intersectCenter)
            # chead1 = cVector+ np.array(intersectCenter)
            # plt.imshow(im5)
            # plt.plot(cRear1[0],cRear1[1],'ro')
            # plt.plot(chead1[0],chead1[1],'go')
            # plt.show()
            # plt.plot([0,50*horizVector[0]],[0,50*horizVector[1]],color='k',lw=5)
            # plt.plot([0,50*vertVector[0]],[0,50*vertVector[1]],color = 'g',lw=5)
            # plt.plot(cVector[0],cVector[1],'ko')
            # a11 = np.array([x_pos-intersectCenter[0],y_pos-intersectCenter[1]])
            # plt.plot(a11[0],a11[1],'go')
            # b11 = a11 - hBL*cToHead
            # plt.plot(b11[0],b11[1],'ro')
            # #plt.plot(a[0]+intersectCenter[0],intersectCenter[1]+a[1],'o')
            # plt.plot(t1[0]-intersectCenter[0],t1[1]-intersectCenter[1],color='r')
            # plt.plot(t2[0]-intersectCenter[0],t2[1]-intersectCenter[1],color='cyan')
            # # cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-hBL*cToHead
            # # hVec_rear = cRear - np.dot(cRear,horizVector)*horizVector
            # # cRear = np.array([x_pos - intersectCenter[0],y_pos -intersectCenter[1]])-hBL*cToHead
            # # vVec_rear = cRear - np.dot(cRear,vertVector)*vertVector
            # # print np.dot(hVec,hVec_rear),np.linalg.norm(hVec_rear),hDist
            # # print np.dot(vVec,vVec_rear),np.linalg.norm(vVec_rear),vDist
            # plt.show()
            tVal = np.arccos(np.dot(cToHead, vertVector))*180./np.pi

            if quad_hot:
                vDist *= -1.

            if quad_hot_rear:
                vDistRear *= -1.

            # if vVec_rear[0]>0 or (vVec_rear[0]<0 and vDistRear<0):
                # tVal = 180.-tVal

            if vDist > hLimit or vDist < lLimit or vDistRear < vDist:
                return 0, 0, vDist

            return 1, tVal, vDist
        else:
            return 0, 0, mDist
    else:
        return 0, 0, mDist


def closerHorV(x_pos, y_pos, theta):
    # np.array([np.cos(np.pi/180*theta),-np.sin(np.pi/180*theta)])
    cToHead = np.array([-np.sin(np.pi/180.*theta), np.cos(np.pi/180.*theta)])
    cVector = np.array([x_pos - intersectCenter[0],
                       y_pos - intersectCenter[1]])+hBL*cToHead
    hVec = cVector - np.dot(cVector, horizVector)*horizVector
    vVec = cVector - np.dot(cVector, vertVector)*vertVector
    hDist = np.linalg.norm(hVec)
    vDist = np.linalg.norm(vVec)
    # print hDist,vDist
    #a = cVector+hBL*cToHead
    # to check geometry
    # print hDist,vDist
    # vid = imageio.get_reader(filename,  'ffmpeg')#for demoing.
    mDist = np.min([hDist, vDist])
    if hDist < vDist:

        return 'h', hDist, vDist
    elif vDist <= hDist:
        return 'v', hDist, vDist


def fullQuadCalc(x_list, y_list):
    quadNum = np.zeros(len(x_list))
    for i in range(0, len(x_list)):
        quadNum[i] = calcQuadrant(y_list[i], x_list[i])

    return quadNum


def calcThreshold(filename):
    # this function performs the data extraction from the video, converting the fly to a binary image
    # and then performing a least squares ellipse fit to find the orientation of the fly.
    # A few different centroid calculations can be done. At the moment, the centroid is taken as the
    # center of mass of the fly. This will be updated soon however.
    import imageio
    import cv2
    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    vDims = vid.get_meta_data()['source_size']
    vLength = vid.get_length()
    nStart = 0
    nFinish = vLength
    nums = range(nStart, nFinish)

    allThetas = np.array([])
    allCenters = np.array([])
    allCOM = np.array([])
    al1s = np.array([])
    al2s = np.array([])
    listThetas = []
    listCenters = []
    listCOM = []
    listAL1 = []
    listAL2 = []
    # cDir = theta
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frames = []
    badCount = 0
    cutoffThresh = 30
    # print vDims
    # print kernel
    mask = np.ones(vDims, dtype="uint8")*255
    for i in range(0, vDims[0]):
        for j in range(0, vDims[1]):
            if np.sqrt((q4[0]-i)**2 + (q4[1]-j)**2) > ((halfWidth + halfHeight)/2+2):
                mask[i, j] = 0

    a1, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # mask.tolist()
    # just need to make sure that the first frame is a good shot... otherwise there can be a problem.
    for threshVal1 in np.arange(50, 100, 5):
        list_pxs = []
        for num in nums:
            im5 = vid.get_data(num)
            # if num ==0:
            # 	plt.imshow(im5)
            imageG5_1 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
            # imageG5_1 = cv2.bitwise_and(imageG5,imageG5,mask=mask)#[25:20*scaling/4.70,25:20*scaling/4.70]
            r1, t5_1 = cv2.threshold(
                imageG5_1, threshVal1, 255, cv2.THRESH_BINARY)
            # tOld = t5_1
            # [25:20*scaling/4.70,25:20*scaling/4.70]
            t5_1 = cv2.bitwise_and(mask.T, t5_1, t5_1)
            # asdfas
       #t5_1 = cv2.dilate(t5_1,kernel,iterations=1)

            t5_1 = cv2.erode(t5_1, kernel, iterations=1)
            t5_1 = cv2.dilate(t5_1, kernel, iterations=1)
            tOld = t5_1
            t5_1 = cv2.bitwise_not(t5_1)
            # t5_1 = cv2.erode(t5_1,kernel,iterations = 1)
            # t5_1 = cv2.dilate(t5_1,kernel,iterations = 1)
            # cv2.imshow("BLA",t5_1)
            # cv2.waitKey(50)
            list_pxs.append(np.sum(1-t5_1/255))
        # print np.median(list_pxs),threshVal1
        if np.median(list_pxs) <= cutoffThresh:
            return threshVal1

        # im2, contours,hierarchy = cv2.findContours(t5_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # if len(frames)>1 and num>600:


if __name__ == "__main__":
    import cv2
    import imageio
    import os
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np
    import sys
    import pickle
    import io
    from PIL import Image

    centerType = "COM"

    saveData = True
    suppress = int(sys.argv[7]) # suppress big output
    filename = str(sys.argv[1])
    showQuadrants = int(filename.split("/")[-1].split(".")[0].split("_")[-1])
    # flipping
    #showQuadrants = 3-showQuadrants
    toPlot = int(sys.argv[2])
    arenaFile = str(sys.argv[3])
    outputFolder = str(sys.argv[4])
    threshVal1 = float(sys.argv[5])
    scaling = float(sys.argv[6])
    bodyLength = scaling*3.0
    # bodyLength = float(sys.argv[7])
    hBL = .5*bodyLength
    antennaeDist = scaling*0.3
    # antennaeDist = float(sys.argv[8])
    # filename = vidDir + "/" +fname #filenames[1]
    data = pickle.load(open(arenaFile, "rb"), encoding="bytes")
    q4 = list(data[0])
    halfWidth = data[1]/2.
    halfHeight = data[2]/2.
    hh1 = (halfWidth+halfHeight)/2.
    halfWidth = hh1
    halfHeight = hh1
    # print halfWidth,halfHeight
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

    # Parameters to study wall boundary interactions
    nearThresh = hBL
    from scipy.interpolate import griddata
    import pickle

    if (filename.split('/')[-1]).split('_')[1] == '25vs25' and 0:
        hLimit = 0
        lLimit = 0
    else:
        (yvals30, levels30, yvals35, levels35, yvals40, levels40, x2, y2, ti, ti35,
         ti0) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")

        if (filename.split('/')[-1]).split('_')[1] == '25vs40':
            ti1 = ti0
        elif (filename.split('/')[-1]).split('_')[1] == '25vs30':
            ti1 = ti
        elif (filename.split('/')[-1]).split('_')[1] == '25vs35':
            ti1 = ti35
        else:
            ti1 = ti0  # this is just an ad hoc fix, so that it runs. temp is not being used for the cold calculations

        # get distance at which temp threshold is reached (25.5)

        threshMax = griddata(ti1[1, :], x2, 25.5)
        print('threshmax=', threshMax)

        hLimit = threshMax  # 12/4.7*scaling
        lLimit = threshMax+5.  # -20/4.7*scaling

        hLimit *= -1*scaling
        lLimit *= -1*scaling

    # parameters for different temperatures at boundary
    # if (filename.split('/')[-1]).split('_')[1] == '25vs40':
    # 	hLimit = 10*scaling/4.7
    # 	lLimit = -10*scaling/4.7
    # elif (filename.split('/')[-1]).split('_')[1] == '25vs35':
    # 	hLimit = 10*scaling/4.7
    # 	lLimit = -10*scaling/4.7
    # elif (filename.split('/')[-1]).split('_')[1] == '25vs30':
    # 	hLimit = 10*scaling/4.7
    # 	lLimit = -10*scaling/4.7
    # else:
    # 	hLimit = 0
    # 	lLimit = 0

    nearThresh_temp = np.max(np.abs([hLimit, lLimit]))

    tH1 = 180./np.pi*np.arctan2(horizVector[1], horizVector[0])
    tV1 = 180./np.pi*np.arctan2(vertVector[1], vertVector[0])
    tV1 = tV1 % 360.
    tH1 = tH1 if abs(tH1) < (90.) else tH1 - 180.

    if threshVal1 == -1:
        threshVal1 = calcThreshold(filename)
        print(threshVal1)
        print(filename)

    # print threshVal1
    # asdfas
    # extract positional/rotational information, plus associated velocities.
    (COM_x1, COM_y1, vx_1, vy_1, denoisedThetas, denoisedThetas_deriv, nStart,
     nFinish, al1s, al2s, frames, quad, qH, qB) = coreDataExtract(filename)
    # now decompose velocity along the direction of the fly's major axis.
    (transV, slipV) = decomposeVelocity(vx_1, vy_1, denoisedThetas)
    (redo, denoisedThetas) = checkOrientation(transV, denoisedThetas)
    if redo == 1:
        print(redo)
        (transV, slipV) = decomposeVelocity(vx_1, vy_1, denoisedThetas)
    nums = range(0, nFinish-nStart)
    # if most translation is backward, rotate theta by 180 degress and repeat velocity decomp.

    (fc_x, fc_y) = formFullCenterList(COM_x1, COM_y1, frames)
    fAngs = formFullAngleList(denoisedThetas, frames)
    fAngs_deriv = formFullAngleList(denoisedThetas_deriv, frames)
    tV = formFullAngleList(transV, frames)
    sV = formFullAngleList(slipV, frames)

    (mode, modeQuality, onWall, tempB, tempB_dist) = timepointClassify(
        transV, denoisedThetas_deriv, al1s, al2s, COM_x1, COM_y1, frames)
    # here, perform modified mode determination.

    head_x = fc_x+hBL*np.cos(np.pi/180*fAngs)
    head_y = fc_y-hBL*np.sin(np.pi/180.*fAngs)
    back_x = fc_x-hBL*np.cos(np.pi/180*fAngs)
    back_y = fc_y+hBL*np.sin(np.pi/180.*fAngs)
    quad_full = fullQuadCalc(fc_x, fc_y)
    quad_head_full = fullQuadCalc(head_x, head_y)
    quad_back_full = fullQuadCalc(back_x, back_y)
    hvCloser = [closerHorV(fc_y[i2], fc_x[i2], fAngs[i2])[0]
                for i2 in range(0, len(head_x))]
    # quad_full = formFullAngleList(quad,frames)
    # quad_full = quadrantFillInMissing(quad_full)
    cold, hot = quadrantTimeSpent(quad_full)

    hcQuads = np.array([(1 if quad_full[j] == 2 or quad_full[j] == 4 else 0 if quad_full[j]
                       == 1 or quad_full[j] == 3 else -1) for j in range(0, len(quad_full))])
    # print len(quad_head_full)
    # quad_head_full = formFullAngleList(qH,frames)
    # quad_head_full = quadrantFillInMissing(quad_head_full)

    # quad_back_full = formFullAngleList(qB,frames)
    # quad_back_full = quadrantFillInMissing(quad_back_full)

    if showQuadrants == 2:
        swap1 = hot
        hot = cold
        cold = swap1
        hcQuads = np.logical_not(hcQuads)

    hcQuads[hcQuads == -1] = 0

    # tV2 = [np.sqrt(tV[j]**2 + sV[j]**2) for j in range(0,len(sV))]
    allEvents = eventTabulate(mode, nStart, nFinish, onWall,
                              quad_full, tempB, tempB_dist, quad_head_full, quad_back_full)
    allEvents = removeShortEvents(allEvents)
    allEvents = calcDistTraveled(allEvents, fc_x, fc_y)

    eventType = eventClassify(allEvents, fc_x, fc_y)
    eventType2 = eventClassify_withLength(allEvents, fc_x, fc_y)

    # xs,ys = np.meshgrid(np.linspace(0,256,256),np.linspace(0,256,256))
    # b = np.zeros(xs.shape)
    # for j0 in range(0,b.shape[0]):
    # 	for k0 in range(0,b.shape[1]):
    # 		if isNearTempBarrier(xs[j0,k0],ys[j0,k0],40.1,10)[0]:
    # 			# print "hiiii"
    # 			b[j0,k0] = isNearTempBarrier(xs[j0,k0],ys[j0,k0],40.1,10)[1]
    # 		# else:
    # 		# 	# print "nothiii"

    # plt.imshow(b)
    # plt.show()

    if toPlot == 1:
        if 1:
            plotWithTurnAngles()
        else:
            plotWithTurnAngles_vidsave()
    elif toPlot == 2:
        plotVelocities()

    if saveData == True:
        trajData = (allEvents, tV, sV, fAngs_deriv, fc_x, fc_y, fAngs,
                    hvCloser, threshVal1, scaling, bodyLength, antennaeDist)
        outFile = outputFolder + '/' + \
            filename.split(".")[0].split("/")[-1] + ".output"
        pickle.dump(trajData, open(outFile, "wb"))

        trajData = (eventType2, [hot, cold])
        outFile = outputFolder + '/' + \
            filename.split(".")[0].split("/")[-1] + ".stats"
        pickle.dump(trajData, open(outFile, "wb"))

    print("===Results===")
    print("Frames of resting:", eventType2[0])
    print("Frames of forward moving left turns:", eventType2[1])
    print("Frames of forward moving right turns:", eventType2[2])
    print("Frames of backward moving left turns:", eventType2[3])
    print("Frames of backward moving right turns:", eventType2[4])
    print("Frames of pure forward moves:", eventType2[5])
    print("Frames of pure backward moves:", eventType2[6])
    print("Frames of stationary left turns:", eventType2[7])
    print("Frames of stationary right turns:", eventType2[8])
    print("Frames in hot regions:", hot)
    print("Frames in cold regions", cold)

    time = np.linspace(0, 3600, 3600)
    domain2plot = range(0, 3600)
    pltZeros = np.zeros(len(domain2plot))
    if 0:
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(time[domain2plot], tV[domain2plot])
        ax[1].plot(time[domain2plot], sV[domain2plot])
        ax[2].plot(time[domain2plot], fAngs_deriv[domain2plot])
        ax[3].fill_between(time[domain2plot], hcQuads[domain2plot])
        for i in range(0, len(allEvents)):
            s, f = allEvents[i].getStartAndFinish()
            # print s,f,allEvents[i].getModes()
            if s >= domain2plot[0] and f <= (domain2plot[-1]+1):
                cM = (allEvents[i].getModes()).astype(int)

                if -1 not in set(cM):

                    # print num,s1,f1,cM,(2 in set(cM) or 8 in set(cM))
                    if (1 in set(cM)):
                        # print "Moving Left turn"
                        col1 = 'g'
                    elif (2 in set(cM)):
                        # print "Moving Right turn"
                        col1 = 'b'
                    elif (3 in set(cM)):
                        # print "back left"
                        col1 = 'purple'

                    elif (4 in set(cM)):
                        # print "back right"
                        col1 = 'y'
                    elif (7 in set(cM)):
                        # print "Stationary Left turn"
                        col1 = 'olive'
                    elif (8 in set(cM)):
                        # print "Stationary Right turn"
                        col1 = 'cyan'
                    elif (5 in set(cM)):
                        # print "forward"
                        col1 = 'r'
                    elif (6 in set(cM)):
                        col1 = 'orange'
                    elif(0 in set(cM)):
                        col1 = 'k'
                    else:
                        print(cM)
                        col1 = 'white'

                    ax[0].axvspan(s, f+1, alpha=0.5, color=col1)
                    ax[1].axvspan(s, f+1, alpha=0.5, color=col1)
                    ax[2].axvspan(s, f+1, alpha=0.5, color=col1)

        ax[0].set_ylabel('trans. vel')
        ax[1].set_ylabel('slip. vel')
        ax[2].set_ylabel('rot. vel')
        ax[3].set_ylabel('in heat')
        ax[0].plot(time[domain2plot], pltZeros, color='k')
        ax[1].plot(time[domain2plot], pltZeros, color='k')
        ax[2].plot(time[domain2plot], pltZeros, color='k')
        plt.show()

        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        jOut = [tV, sV, fAngs_deriv, fc_x, fc_y, fAngs]
        jDat = json.dumps(jOut, cls=NumpyEncoder)
        with open('tracking_data.json', 'w') as f:
            json.dump(jDat, f)
>>>>>>> Stashed changes
