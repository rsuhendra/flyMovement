import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
import os
import io
from PIL import Image

def calcQuadrant1(x_pos, y_pos):

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

def isQuadHot1(quad,state):
	if state==1:
		return quad ==2.0 or quad==4.0
	else:
		return quad ==1.0 or quad == 3.0

def calc_distance(cVector,quad_hot):
	hVec = cVector - np.dot(cVector,horizVector)*horizVector
	vVec = cVector - np.dot(cVector,vertVector)*vertVector
	hDist = np.linalg.norm(hVec)
	vDist = np.linalg.norm(vVec)

	mDist = np.min([hDist,vDist])
	# if mDist < 20.:
	if hDist<vDist:

		if quad_hot:
			hDist*=-1. 

		return hDist

	elif vDist<=hDist:

		if quad_hot:
			vDist*=-1. 

		return vDist
	# else:
	# 	print 'theres an issue',mDist

# dataFname = "./output/"

(yvals30,levels30,yvals35,levels35,yvals40,levels40,x2,y2,ti,ti35,ti0) = pickle.load(open("contour_info30_40.pkl","rb"),encoding='latin1')
tempTest = 1
inputF = str(sys.argv[1])
groupName = 'DNB05+'
inputDir = 'outputs/outputs_'+groupName+'/output_40/'
plotdir = 'polar/polar_plots/polar_plot_'+groupName+'/'
datadir = 'polar/polar_data/polar_data_'+groupName+'/'

if not os.path.exists(plotdir):
	os.makedirs(plotdir)
if not os.path.exists(datadir):
	os.makedirs(datadir)

(inThetas,outThetas,decTime,eventNum,eventType,l,listFNames,eventFrames,eventFNames) = pickle.load(open(inputF,"rb"))
eventNum = [eventNum[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]
inThetas = [inThetas[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]
outThetas = [outThetas[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]
decTime = [decTime[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]

e_frames = [eventFrames[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]
e_fnames = [eventFNames[i] for i in range(0,len(eventNum)) if eventType[i]=='turn' ]

if tempTest:
	listEventNum,angList,ang_val_list = [],[],[]
	for filename in os.listdir(inputDir):
		if filename.split(".")[-1] == 'output':
			(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,scaling,bodyLength,antennaeDist) = pickle.load(open(inputDir+filename,"rb"))
			#extract max event size. 
			for l in range(0,len(allEvents)):
				s1,f1 = allEvents[l].getStartAndFinish()
				cM = (allEvents[l].getModes()).astype(int)

				if allEvents[l].onWall !=3 and fc_y[s1]!=0.:# and inHot>0 and inCold>0 :
					if ((f1-s1)>1):
						if (-1 not in set(cM)):
							angInit = allEvents[l].nearTempBarrier
							ang_val = fAngs[s1:f1+1]
							listEventNum.append(l)
							angList.append(angInit)
							ang_val_list.append(ang_val)

###open file, determine quadrant
from scipy.interpolate import griddata
tDiff = []
tDiff_fnames = []
for i in range(0,len(e_fnames)):
	filename1 = e_fnames[i]
	if (filename1.split('/')[-1]).split('_')[1] == '25vs40' or (filename1.split('/')[-1]).split('_')[1] == '25vs25':
		ti1 = ti0
	elif (filename1.split('/')[-1]).split('_')[1] == '25vs30':
		ti1 = ti
	elif  (filename1.split('/')[-1]).split('_')[1] == '25vs35':
		ti1 = ti35

	(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,scaling,bodyLength,antennaeDist) = pickle.load(open(inputDir+filename1,"rb"))
	showQuadrants = int(filename1.split(".")[0].split("_")[-1])
	frames1 = e_frames[i]
	arenaFile = allEvents[0].arenaFile
	aF = open(arenaFile,"rb")
	data = pickle.load(aF)
	q4 = list(data[0])
	halfWidth = data[1]/2.
	halfHeight = data[2]/2.
	t1 = np.array(data[3])
	t2 = np.array(data[4])
	aF.close()
	hBL = 0.5*bodyLength
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



	#hvcloser
	hv1 = hvCloser[frames1[0]:frames1[1]][0]
	#quadrant
	hc1 = calcQuadrant1(fc_y[frames1[0]]-hBL*np.sin(np.pi/180.*fAngs[frames1[0]]),fc_x[frames1[0]]+ hBL*np.cos(np.pi/180.*fAngs[frames1[0]]))
	# print hc1
	# print showQuadrants
	#is it hot. 
	hotQ1 = isQuadHot1(hc1,showQuadrants)
	# print hotQ1

	if hv1=='v' and ((hc1==2 and not hotQ1) or (hc1==3 and not hotQ1) or (hc1==1 and hotQ1) or (hc1==4 and hotQ1)):
		inThetas[i]= 180.-inThetas[i]

	if hv1=='h' and ((hc1==3 and not hotQ1) or (hc1==4 and not hotQ1) or (hc1==1 and hotQ1) or (hc1==2 and hotQ1)):
		inThetas[i]= 180.-inThetas[i]



	############calculate distances to left and right antennae, for calculating temp differences between them. 
	if tempTest:
		#get first frame at which head is past the 25.5 line. 
		# print'ha', allEvents[listEventNum[eventNum[i]]].frameEntryT
		i0 = allEvents[listEventNum[eventNum[i]]].frameEntryT-1
		print (i0, frames1,fAngs[i0],inThetas[i])
		# cToHead = np.array([-np.sin(np.pi/180.*theta),np.cos(np.pi/180.*theta)])

		cToRightAnt = [fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) -antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])-antennaeDist*np.sin(np.pi/180.*fAngs[i0])]
		cToLeftAnt = [fc_y[i0]-hBL*np.sin(np.pi/180.*fAngs[i0]) +antennaeDist*np.cos(np.pi/180.*fAngs[i0]),fc_x[i0]+ hBL*np.cos(np.pi/180.*fAngs[i0])+antennaeDist*np.sin(np.pi/180.*fAngs[i0])]
		

		cToLeftAnt = [cToLeftAnt[0]-intersectCenter[0],cToLeftAnt[1]-intersectCenter[1]]
		cToRightAnt = [cToRightAnt[0]-intersectCenter[0],cToRightAnt[1]-intersectCenter[1]]

		dForLeftAnt = -1*calc_distance(cToLeftAnt,hotQ1)/scaling
		dForRightAnt =  -1*calc_distance(cToRightAnt,hotQ1)/scaling

		# # for checking theta. 
		angle1 = (np.pi/2. - np.arcsin(np.abs(dForRightAnt-dForLeftAnt)/0.3))*180./np.pi
		print (angle1,hotQ1,hc1,hv1)
		if dForLeftAnt<dForRightAnt:
			angle1= 180. - angle1

		print (angle1,inThetas[i],'hi', dForRightAnt-dForLeftAnt,dForLeftAnt, dForRightAnt )

		# inThetas[i] = angle1

		tempLeftAnt = griddata(x2,ti1[1,:],dForLeftAnt)
		tempRightAnt = griddata(x2,ti1[1,:],dForRightAnt)
		# print 180.-inThetas[i],dForLeftAnt,dForRightAnt
		# print 180.-inThetas[i],tempLeftAnt,tempRightAnt
		tDiff.append(tempLeftAnt-tempRightAnt)
		tDiff_fnames.append(filename1)
	# asdfs

inThetas = [180. - inThetas[j] for j in range(0,len(inThetas))]
fig,ax=plt.subplots()
scat2 =ax.scatter(inThetas,outThetas,c=np.log(np.array(decTime)+1),picker=5,s=10)
plt.colorbar(scat2,ax=ax)
ax.set_xlabel('initial angle relative to temp wall')
ax.set_ylabel('resulting turn angle')
ax.set_xlim([0,180])
# fig.canvas.mpl_connect('pick_event', onpick_temp_boundary)
plt.savefig(plotdir+'input_output_'+inputF.split('.')[0]+'.png')

groupSmall = []
groupLarge = []
for i in range(0, len(inThetas)):
	if inThetas[i]<40 or inThetas[i]>140:
		groupSmall.append(decTime[i])
	else:
		groupLarge.append(decTime[i])


# print np.mean(groupSmall), np.mean(groupLarge),np.std(groupSmall), np.std(groupLarge)

# from scipy.stats import ttest_ind

# t,p = ttest_ind(groupSmall,groupLarge,equal_var=False)
# print t,p

color = outThetas
fig, ax = plt.subplots()
n1 = np.round(len(inThetas)/4.).astype('uint8')
n,bins,patches = plt.hist(color,bins=n1,facecolor='b',density=True,alpha=0.75)
plt.grid(True)
plt.savefig(plotdir+'allTurnsHist'+inputF.split('.')[0]+'.png')


#from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

xmin1 = np.min(color)
xmax1 = np.max(color)
dmain= np.linspace(xmin1,xmax1,100)
# grid = GridSearchCV(KernelDensity(),{'bandwidth':np.linspace(-1,20.0,25)},cv=5)
# grid.fit(np.array(color)[:,None])
# bw1 = grid.best_params_.get('bandwidth')
# print bw1, "bw1"
import seaborn as sns
fig, ax = plt.subplots()
sns.kdeplot(np.array(color),bw=5,shade=True)
ax.set_xlim([-250,250])
# tran1 = KernelDensity(bandwidth=3).fit(np.array(color)[:,None])
# lDens1 = tran1.score_samples(dmain[:,None])

# plt.plot(dmain,np.exp(lDens1))
plt.grid(True)

plt.savefig(plotdir+'smoothedVals_selected_'+inputF.split('.')[0]+'.png')
import json
class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)
jOut = [color]
jDat = json.dumps(jOut,cls=NumpyEncoder)
with open(datadir+inputF.split('.')[0]+'_turningVals.json','w') as f:
	json.dump(jDat,f)

bins = np.linspace(0,180,5)
halfPts = (bins[1]-bins[0])/2 + bins[0:len(bins)-1]
inds = np.digitize(inThetas,bins,right=True)
print (inds,bins)
inds = inds-1
numSlots = len(bins)-1
rCount = np.zeros(numSlots)
lCount = np.zeros(numSlots)
sCount = np.zeros(numSlots)
angThres = 15
for i in range(0,len(inThetas)):
	if outThetas[i]<-angThres:
		lCount[inds[i]]+=1

	elif outThetas[i]>angThres:
		rCount[inds[i]]+=1

	else:
		sCount[inds[i]]+=1

rCountNorm = np.zeros(len(rCount))
sCountNorm = np.zeros(len(rCount))
lCountNorm = np.zeros(len(rCount))
for i in range(0,len(rCount)):
	tot = rCount[i]+lCount[i]#+sCount[i]
	rCountNorm[i] =rCount[i]/tot
	# sCountNorm[i]= sCount[i]/tot
	lCountNorm[i] = lCount[i]/tot


import csv
with open(datadir+'special_count_values_'+inputF.split('.')[0]+'.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['Right'])
	writer.writerow(rCount)
	writer.writerow(['Left'])
	writer.writerow(lCount)


import seaborn as sns
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111,projection='polar')
print (halfPts, rCount.shape)
# print np.concatenate((sCountNorm,rCountNorm))
p1= plt.bar(halfPts*np.pi/180.,rCountNorm,width = (bins[1]-bins[0])*np.pi/180.,color='purple',edgecolor='k',linewidth=1.5)
# p2 = plt.bar(halfPts*np.pi/180.,sCountNorm,width = (bins[1]-bins[0])*np.pi/180.,color='grey',bottom=rCountNorm,edgecolor='k',linewidth=1.5)
p3 = plt.bar(halfPts*np.pi/180.,lCountNorm,width = (bins[1]-bins[0])*np.pi/180.,color='green',bottom =rCountNorm,edgecolor='k',linewidth=1.5)
# plt.stackplot(bins,[rCount,sCount,lCount],labels=['Right','Straight','Left'])
plt.legend((p1[0],p3[0]),('Right','Left'),loc='upper right')
plt.xlabel('Incoming angle')
ax.set_rorigin(-1.0)
ax.set_thetamin(0)
ax.set_thetamax(180)
ax.set_ylim([0,1])
ax.xaxis.grid(False)
ax.set_xticks(bins*np.pi/180.)
# ax.set_xticks(ticks)
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig(plotdir+'prob_turn_'+inputF.split('.')[0]+'.pdf')


######
#store data for statistical testing relative to the WT. 
# turnData = [rCount,sCount,lCount]
# outfile= inputF.split('.')[0] + '_turnData.pkl'
plt.close('all')
if tempTest:
	tDeltas = np.array(tDiff)
	groupCut = 0.15
	inds1 = tDeltas<=-groupCut
	inds2 = tDeltas>=groupCut
	inds3 = [tDeltas[i]>-groupCut and tDeltas[i]<groupCut for i in range(0,len(tDeltas))]

	outThetas= np.array(outThetas)
	g1 = outThetas[inds1]
	g2 = outThetas[inds2]
	g3 = outThetas[inds3]
	fig,ax = plt.subplots()
	lab1 = ['<-cut','[-cut,cut]','>cut']
	ax.boxplot([g1,g3,g2],labels=lab1)
	ax.set_xlabel('T difference between antennae at turn start (C)')
	ax.set_ylabel('Rotation angle of turn (deg)')
	plt.savefig(plotdir+'turn_dir_boxplot'+inputF.split('.')[0]+'.pdf')
	plt.close()
	bins = np.linspace(0,180,5)
	aThresh = 15
	print (outThetas[inds1]>aThresh)
	rights1 = np.sum(outThetas[inds1]>aThresh)
	lefts1 = np.sum(outThetas[inds1]<-aThresh)
	rights2 = np.sum(outThetas[inds2]>aThresh)
	lefts2 = np.sum(outThetas[inds2]<-aThresh)
	rights3 = np.sum(outThetas[inds3]>aThresh)
	lefts3 = np.sum(outThetas[inds3]<-aThresh)
	tot1 = rights1+lefts1
	print (tot1)
	tot2 = rights2+lefts2
	tot3 = rights3+lefts3
	try:
		lefts = np.array([1.0*lefts1/tot1,1.0*lefts3/tot3,1.0*lefts2/tot2])
		rights = np.array([1.0*rights1/tot1,1.0*rights3/tot3,1.0*rights2/tot2])
		import seaborn as sns
		print (lefts,rights)
		fig,ax = plt.subplots()
		p2 = ax.bar(np.array([0,1,2]),rights,color = 'purple')
		p1 =ax.bar(np.array([0,1,2]),lefts,bottom=rights,color = 'green')
		ax.set_xticks([0,1,2])
		ax.set_xticklabels(['<-L','[-L,L]','>L'])
		plt.savefig(plotdir+inputF.split('.')[0] + '_turn_dir_fractions.pdf')
	except:
		pass

lefts0 = [lefts1,lefts2,lefts3]
rights0 = [rights1,rights2,rights3]
turnData = [rCount,sCount,lCount,lefts0,rights0]
outfile= datadir+inputF.split('.')[0] + '_turnData.pkl'
pickle.dump(turnData,open(outfile,"wb"))

plt.close()
plt.scatter(tDiff,outThetas)
plt.savefig(plotdir+inputF.split('.')[0] + '_diff_dirplot.pdf')
plt.close()
outfile= datadir+inputF.split('.')[0] + '_grouped_turn_data.pkl'
pickle.dump((inThetas,outThetas,tDiff,tDiff_fnames),open(outfile,"wb"))


# ######### for turn prediction. 

lDiffs = [np.abs(tDiff[i]) for i in range(0,len(tDiff))if tDiff[i]>0. and abs(outThetas[i])>15.]
rDiffs = [np.abs(tDiff[i]) for i in range(0,len(tDiff))if tDiff[i]<0. and abs(outThetas[i])>15.]
lOuts = [outThetas[i]>0. for i in range(0,len(tDiff))if tDiff[i]>0. and abs(outThetas[i])>15.]
rOuts = [outThetas[i]<0. for i in range(0,len(tDiff))if tDiff[i]<0. and abs(outThetas[i])>15.]

tdiff1 = lDiffs+rDiffs
tOuts1 = lOuts+rOuts

bins = np.linspace(0,.5,6)
midPoints = [(bins[i]+bins[i-1])/2. for i in range(1,len(bins))]+ [(bins[-1]+bins[-2])/2.+(bins[-1]-bins[-2])]
inds1 = np.digitize(tdiff1,bins)-1
print (tdiff1)

fig,ax = plt.subplots()
nBootstraps = 1000
allRatios = []
tOuts1 = np.array(tOuts1)
inds1 = np.array(inds1)
for k in range(0,nBootstraps):
	ratios=[]
	bootInds= np.random.randint(0,len(inds1),len(inds1))
	tOuts11 = tOuts1[bootInds]
	inds11 = inds1[bootInds]
	for i in range(0,len(bins)):
		ratios.append(np.mean([tOuts11[j]*1.0 for j in range(0,len(inds11)) if inds11[j]==i]))
	allRatios.append(ratios)
allRatios_STDEV = np.std(allRatios,axis=0)
allRatios_MEAN = np.mean(allRatios,axis=0)
ax.plot(midPoints[0:len(midPoints)-1],allRatios_MEAN[0:len(midPoints)-1],color='r')
ax.fill_between(midPoints[0:len(midPoints)-1],allRatios_MEAN[0:len(midPoints)-1] - allRatios_STDEV[0:len(midPoints)-1],allRatios_MEAN[0:len(midPoints)-1]+allRatios_STDEV[0:len(midPoints)-1],alpha = 0.3,color='r')
xlabs = ['['+str(np.round(bins[i],1))+','+str(np.round(bins[i+1],1))+')' for i in range(0,len(midPoints)-1) ] + ['>'+str(bins[-1])]
ax.set_xticks(midPoints)
ax.set_xticklabels(xlabs)
ax.set_ylim(0,1)
plt.savefig(plotdir+inputF.split('.')[0] + '_bootstrap_turn_pred.pdf')
pickle.dump((inds1,tOuts1),open( datadir+inputF.split('.')[0] +"ratios_tempDiff.pkl", "wb" ))


# #fit this with a logistic-like function. 
# from scipy.optimize import least_squares

# def fun(x1, t1, y1):
# 	t1 = np.array(t1)
# 	return x1[0] + x1[1]/(1.+ np.exp(-x1[2] * t1)) - y1
# x0 = np.array([-50,50,1])


# def f2(x1, t1):
# 	t1 = np.array(t1)
# 	return x1[0] + x1[1]/(1.+ np.exp(-x1[2] * t1))
# tMax = np.max(np.abs(tDiff))
# t_dom = np.linspace(-tMax,tMax,100)

# numResamples = 1000
# confidenceVals = []
# tDiff = np.array(tDiff)
# for j in xrange(0,numResamples):
# 	bootstrapInds = np.random.randint(0,len(tDiff),len(tDiff))
# 	tDiffResample = tDiff[bootstrapInds]
# 	outThetasResample = outThetas[bootstrapInds]
# 	res_lsq = least_squares(fun, x0, args=(tDiffResample,outThetasResample))
# 	# plt.plot(t_dom,f2(res_lsq.x ,t_dom))
# 	confidenceVals.append(f2(res_lsq.x ,t_dom))

# import scipy.stats as st
# print len(tDiff)
# confidenceVals = np.array(confidenceVals)
# locs = np.mean(confidenceVals,axis=0)
# stds = np.std(confidenceVals,axis=0)
# bstrap_ci1 = np.percentile(confidenceVals,2.5,axis=0)
# bstrap_ci2 = np.percentile(confidenceVals,97.5,axis=0)

# plt.scatter(tDiff,outThetas)


# res_lsq = least_squares(fun, x0, args=(tDiff,outThetas))
# plt.plot(t_dom,f2(res_lsq.x ,t_dom))
# print res_lsq.x
# # plt.plot(t_dom,locs)
# plt.fill_between(t_dom,bstrap_ci1,bstrap_ci2,alpha =0.3)
# plt.savefig(inputF.split('.')[0] + '_confidence_dir_pred.pdf')
# # plt.show()

