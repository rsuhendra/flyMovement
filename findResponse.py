import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
import os
from PIL import Image


# dataFname = "./output/"
inputDir = str(sys.argv[1])
# arenaData = pickle.load(open(arenaFile,"rb"))
colorOption = 3
behavior = []
behaviorCat = []
maxDiff = 0

# for filename in os.listdir(inputDir):
# 	(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs) = pickle.load(open(inputDir+filename,"rb"))
# 	#extract max event size. 
# 	for l in xrange(0,len(allEvents)):
# 		s1,f1 = allEvents[l].getStartAndFinish()
# 		cM = (allEvents[l].getModes()).astype(int)
# 		if (0 not in set(cM)):
# 			if 1+f1-s1>maxDiff:
# 				maxDiff = 1+f1-s1
			#behavior.append((tV[s1:f1+1],sV[s1:f1+1],fAngs_deriv[s1:f1+1]))

		# if l>0:
		# 	N = np.min([len(behavior[l-1][0]),len(behavior[l][0])])
		# 	#N = np.min(len(behavior[l][0]),len(behavior[l-1][0]))
		# 	#print distMetric(behavior[l],behavior[l-1])
print (maxDiff)
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

hotQuads = [1,3] 
coldQuads = [2,4]
onW = []
onW_inC = []
k=0
useAll = 0
maxDiff = 300
wallCheck = []
for filename in os.listdir(inputDir):
	if filename.split(".")[-1] == 'output':
		(allEvents,tV,sV,fAngs_deriv,fc_x,fc_y,fAngs,hvCloser,threshVal1,scaling,bodyLength,antennaeDist) = pickle.load(open(inputDir+filename,"rb"))
		#extract max event size. 
		for l in range(0,len(allEvents)):
			s1,f1 = allEvents[l].getStartAndFinish()
			cM = (allEvents[l].getModes()).astype(int)
			quads = allEvents[l].quadrants
			quads_back = allEvents[l].quadrants_back
			quads_head = allEvents[l].quadrants_head
			inHot = np.sum([quads1 in hotQuads for quads1 in quads])
			inCold = np.sum([quads1 in coldQuads for quads1 in quads])
			arenaFile = allEvents[l].arenaFile
			showQuadrants = allEvents[l].showQuadrants
			# @Miguel Comment out the line below for older arena configuration, comment in for newer ones
			showQuadrants = 3 - showQuadrants
			if showQuadrants ==1:
				startInCold = quads[0] in hotQuads
				startInCold_head = quads_head[0] in hotQuads
				startInCold_back = quads_back[0] in hotQuads
			else:
				startInCold = quads[0] in coldQuads
				startInCold_head = quads_head[0] in coldQuads
				startInCold_back = quads_back[0] in coldQuads
			maxProj = np.asarray(Image.open(allEvents[l].maxProj).convert('L'))

			if allEvents[l].onWall !=3 and fc_y[s1]!=0.:# and inHot>0 and inCold>0 :
				if ((f1-s1)>1):
					if (f1-s1)<maxDiff:			
						L = maxDiff -(f1-s1)-1
						tv_pad = np.lib.pad(tV[s1:f1+1],(0,L),'constant',constant_values=(0,0))
						sv_pad = np.lib.pad(sV[s1:f1+1],(0,L),'constant',constant_values=(0,0))
						angs_pad = np.lib.pad(fAngs_deriv[s1:f1+1],(0,L),'constant',constant_values=(0,0))
					else:
						tv_pad = tV[s1:s1+maxDiff]#np.lib.pad(tV[s1:f1+1],(0,L),'constant',constant_values=(0,0))
						sv_pad = sV[s1:s1+maxDiff]#np.lib.pad(sV[s1:f1+1],(0,L),'constant',constant_values=(0,0))
						angs_pad = fAngs_deriv[s1:s1+maxDiff]#np.lib.pad(fAngs_deriv[s1:f1+1],(0,L),'constant',constant_values=(0,0))
					angCov = fAngs[f1]-fAngs[s1]
					angInit = allEvents[l].nearTempBarrier
					if np.all(fc_y[np.max([0,s1-5]):np.min([f1+5,3600])]) and 0:
						pos = (fc_y[np.max([0,s1-5]):np.min([f1+5,3600])],fc_x[np.max([0,s1-5]):np.min([f1+5,3600])])
						ang_val = fAngs[np.max([0,s1-5]):np.min([f1+5,3600])]
					else:
						pos = (fc_y[s1:f1+1],fc_x[s1:f1+1])
						ang_val = fAngs[s1:f1+1]

					if (-1 not in set(cM)):
						if angInit !=(-180):
							onW.append(1)
						else:
							onW.append(0)
						if angInit !=(-180):
							onW_inC.append(1)
						else:
							onW_inC.append(0)
						wallCheck.append(allEvents[l].onWall)
					if colorOption ==3:
						if (-1 not in set(cM)):
							maxProjList.append(maxProj)
							quadList.append(showQuadrants)
							behavior.append([tv_pad,sv_pad,angs_pad])
							behaviorCat.append(np.concatenate([tv_pad,sv_pad,angs_pad]))
							posList.append(pos)
							color.append(np.sum(angs_pad))
							list1.append((s1,f1))
							listSF.append(f1-s1)
							listFNames.append(filename)
							listEventNum.append(l)
							angList.append(angInit)
							ang_val_list.append(ang_val)
							af_list.append(arenaFile)

							if allEvents[l].frameEntryT != -1:
								if (f1 - allEvents[l].frameEntryT+1) >=0:
									listFramesInTempB.append(f1-allEvents[l].frameEntryT+1)
									# print s1,f1,allEvents[l].frameEntryT
								else:
									listFramesInTempB.append(0)
							else:
								listFramesInTempB.append(0)



print (len(color),len(behavior))
from sklearn.manifold import TSNE,Isomap,LocallyLinearEmbedding
from sklearn.decomposition import PCA
from matplotlib.patches import Wedge,Circle


# def onpick_temp_boundary(event):
# 	thisline = event.artist
# 	ind = event.ind
# 	cL = newListemb[int(ind[0])]
# 	print('onpick points:', int(ind[0]),color[cL])
# 	fsLength = listSF[cL]
# 	title = listFNames[cL]
# 	evNum = listEventNum[cL]
# 	fig2= plt.figure(figsize=(15,15))
# 	ax3 = fig2.add_subplot(1,4,1)
# 	transVs = behavior[cL][0][0:fsLength]
# 	ax3.plot(transVs)
# 	ax3.set_ylabel('translational velocity')
# 	plt.title(title+" at number: "+str(evNum))
# 	ax4 = fig2.add_subplot(1,4,2)
# 	ang1s = behavior[cL][2][0:fsLength]
# 	ax4.plot(ang1s)
# 	ax4.set_ylabel('angular velocity')
# 	ax4.set_ylim((np.min(np.concatenate(([-4],ang1s))),np.max(np.concatenate(([4],ang1s)))))
# 	aF = open(af_list[cL],"rb")
# 	data = pickle.load(aF)
# 	q4 = list(data[0])
# 	halfWidth = data[1]/2.
# 	halfHeight = data[2]/2.
# 	aF.close()
# 	ax5 = fig2.add_subplot(1,4,3)
# 	for g in groups:
# 		if cL in set(g):
# 			print g
# 			# g.insert(0,g[-1]-1)
# 			# g.append(g[-1]+1)
# 			# g.append(g[-1]+2)
# 			fullPoslist = np.concatenate([posList[c2] for c2 in g],axis=1)

# 	pos1 = posList[cL]
# 	hTheta = np.array(ang_val_list[cL])
# 	head_pos = np.array(pos1)
# 	head_pos[0,:] = head_pos[0,:] - 7.0*np.sin(np.pi/180*hTheta)
# 	head_pos[1,:] = head_pos[1,:] + 7.0*np.cos(np.pi/180*hTheta)
# 	#head_pos = pos1 + 7.0*[np.cos(hTheta),np.sin(hTheta)]
# 	t1 = data[3]
# 	t2 = data[4]

# 	vertVector = np.array([(t1[0][1]-t1[0][0]),(t1[1][1]-t1[1][0])])
# 	vertVector = vertVector/np.linalg.norm(vertVector)

# 	if vertVector[1]<0:
# 		vertVector = -vertVector

# 	horizVector = np.array([(t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0])])
# 	horizVector = horizVector/np.linalg.norm(horizVector)
# 	if horizVector[0]<0:
# 		horizVector = -horizVector

# 	# angVert = np.arctan2((t1[1][1]-t1[1][0]),(t1[0][1]-t1[0][0]))
# 	# angHoriz = np.arctan2((t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0]))

# 	angVert =np.arctan2(vertVector[1],vertVector[0])
# 	angHoriz = np.arctan2(horizVector[1],horizVector[0])
# 	angVert = (180*angVert/np.pi) % 360.
# 	angVert = 180. - angVert # since image has inverted y axis. 
# 	angHoriz = 180.*angHoriz/np.pi if abs(angHoriz) < (np.pi/2) else 180.*angHoriz/np.pi -180.
# 	angHoriz*=-1.

# 	if quadList[cL] ==1:
# 		topRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,angHoriz,90+angVert,alpha =0.3,color = 'orange')
# 		bottomLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,180+angHoriz,270+angVert,alpha =0.3,color = 'orange')
# 		ax5.add_patch(topRightQuad)
# 		ax5.add_patch(bottomLeftQuad)
# 	elif quadList[cL] ==2:
# 		topLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,90+angVert,180+angHoriz,alpha =0.3,color = 'orange')
# 		bottomRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,270+angVert,360+ angHoriz,alpha =0.3,color = 'orange')
# 		ax5.add_patch(topLeftQuad)
# 		ax5.add_patch(bottomRightQuad)
# 	#(x1s,y1s) = calcTrajectory(transVs,ang1s)
# 	edge = Circle(q4,radius=(halfWidth+halfHeight)/2,color='black',fill=False)
# 	ax5.add_patch(edge)
# 	ax5.plot(fullPoslist[0],fullPoslist[1])

# 	ax5.plot(pos1[0],pos1[1])
# 	ax5.plot(head_pos[0,:],head_pos[1,:],color='red')

# 	ax5.arrow(pos1[0][-1],pos1[1][-1],2*(pos1[0][-1]-pos1[0][-2]),2*(pos1[1][-1]-pos1[1][-2]),head_width=2.0)
# 	ax5.set_aspect('equal')
# 	aLim = ax5.get_ylim()
# 	ax5.set_ylim([aLim[1],aLim[0]])
# 	#ax5.set_xlim((np.min(np.concatenate(([-10],x1s))),np.max(np.concatenate(([10],x1s)))))
# 	# ax5.set_ylim((np.min(np.concatenate(([-10],y1s))),np.max(np.concatenate(([10],y1s)))))
# 	ax5.set_title("trajectory")
# 	ax6 = fig2.add_subplot(1,4,4)
# 	ax6.imshow(maxProjList[cL])
# 	plt.show()


X_emb =[]
Y_emb =[]
listEmb = []
list_inC = []
# print len(onW)
# print len(color)
color_emb = []
for k1 in range(0,len(onW)):
	if onW[k1] ==1:
		listEmb.append(k1)
	if onW_inC[k1]==1:
		list_inC.append(k1)

# print np.sum(onW)


# fig,ax = plt.subplots()
# for l in xrange(0,len(onW)):
# 	head_pos = np.array(posList[l])
# 	hTheta = np.array(ang_val_list[l])

# 	head_pos[0,:] = head_pos[0,:] - 7.0*np.sin(np.pi/180*hTheta)
# 	head_pos[1,:] = head_pos[1,:] + 7.0*np.cos(np.pi/180*hTheta)
# 	if onW_inC[l]==1:
# 		ax.plot(head_pos[0,:],head_pos[1,:],color='red')
# 		ax.plot(posList[l][0],posList[l][1],color='b')
# 	else:
# 		ax.plot(head_pos[0,:],head_pos[1,:],color='k')
# 		ax.plot(posList[l][0],posList[l][1],color='g')

# 	aF = open(af_list[l],"rb")
# 	data = pickle.load(aF)
# 	q4 = list(data[0])
# 	halfWidth = data[1]/2.
# 	halfHeight = data[2]/2.
# 	aF.close()
# 	edge = Circle(q4,radius=(halfWidth+halfHeight)/2,color='black',fill=False)
# 	t1 = data[3]
# 	t2 = data[4]
# 	ax.add_patch(edge)

# 	angVert = np.arctan2((t1[1][1]-t1[1][0]),(t1[0][1]-t1[0][0]))
# 	angHoriz = np.arctan2((t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0]))


# 	if quadList[l] ==1:
# 		topRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,-angHoriz,90-angVert,alpha =0.3,color = 'orange')
# 		bottomLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,180-angHoriz,270-angVert,alpha =0.3,color = 'orange')
# 		ax.add_patch(topRightQuad)
# 		ax.add_patch(bottomLeftQuad)
# 	elif quadList[l] ==2:
# 		topLeftQuad = Wedge(q4,(halfWidth+halfHeight)/2,90-angVert,180-angHoriz,alpha =0.3,color = 'orange')
# 		bottomRightQuad = Wedge(q4,(halfWidth+halfHeight)/2,270-angVert,360- angHoriz,alpha =0.3,color = 'orange')
# 		ax.add_patch(topLeftQuad)
# 		ax.add_patch(bottomRightQuad)
# 	ax.set_aspect('equal')


# plt.show()



group = []
groups = []
groupFnames = []
# print list_inC
for k1 in range(0,len(list_inC)):
	cVal = list_inC[k1]

	if len(group)>0:
		if (cVal-list_inC[k1-1]) ==1 and listFNames[list_inC[k1-1]] ==listFNames[list_inC[k1]]:
		# group.append(listEmb[k1-1])
			group.append(cVal)
		else:
			groups.append(group)
			groupFnames.append(listFNames[list_inC[k1-1]])
			group=[cVal]
	else:
		group.append(cVal)

if len(group)>0:
	groups.append(group)
	groupFnames.append(listFNames[list_inC[k1]])
# print groups

# for i in xrange(0,len(groups)):
# 	if listFNames[groups[i][len(groups[i])-1]]== listFNames[groups[i][len(groups[i])-1]+1] and wallCheck[groups[i][len(groups[i])-1]+1]==0:
# 		groups[i].append(groups[i][len(groups[i])-1]+1)
# # fig,ax = plt.subplots()
colors_list = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059","#FF5733","#FFD133",
	"#FFDBE5", "#7A4900", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
	"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
	"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
	"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
	"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
	"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
	"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
	"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
	"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
	"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
	"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
	"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]


colors_list = [tuple(map(lambda x: x/255.0,bytearray.fromhex(col1[1:7]))) for col1 in colors_list]
inThetas = []
outThetas = []
decTime = []
eventNum = []
eventType = []
eventFrames,eventFNames = [],[]
i= 1
hBL = bodyLength /2. 
print (len(groups))
for g1 in groups:
	# print listFNames[g1]

	lastEnd = list1[g1[0]][1]
	# if list1[g1[0]][0]<300:
	# 	i+=1
	# 	continue
	fig,ax = plt.subplots()

	for i1 in range(0,len(g1)):
		l = g1[i1]
		print (list1[g1[i1]])
		print (angList[g1[i1]])
		if list1[g1[i1]][0] -lastEnd >5 and i1!=0:
			lastEnd = list1[g1[i1]][1]

			print ("check1")
		else:
			lastEnd = list1[g1[i1]][1]
		head_pos = np.array(posList[l])
		hTheta = np.array(ang_val_list[l])

		head_pos[0,:] = head_pos[0,:] - hBL*np.sin(np.pi/180*hTheta)
		head_pos[1,:] = head_pos[1,:] + hBL*np.cos(np.pi/180*hTheta)
		if onW_inC[l]==1:
			try:
				ax.plot(head_pos[0,:],head_pos[1,:],color=colors_list[i1])
				ax.plot(posList[l][0],posList[l][1],color='b')
			except:
				break
		else:
			ax.plot(head_pos[0,:],head_pos[1,:],color='k')
			ax.plot(posList[l][0],posList[l][1],color='g')
		ax.plot(head_pos[0,head_pos.shape[1]-1],head_pos[1,head_pos.shape[1]-1],'o', color='k')

		aF = open(af_list[l],"rb")
		data = pickle.load(aF)
		q4 = list(data[0])
		halfWidth = data[1]/2.
		halfHeight = data[2]/2.
		aF.close()
		t1 = data[3]
		t2 = data[4]
		ax.plot((t1[0][0],t1[0][1]),(t1[1][0],t1[1][1]),color ='g')
		ax.plot((t2[0][0],t2[0][1]),(t2[1][0],t2[1][1]),color ='g')
		# print head_pos[:,0]

		m1 = (t1[0][1]-t1[0][0])/(t1[1][1]-t1[1][0]) # 1/slope

		m2 = (t2[1][1]-t2[1][0])/(t2[0][1]-t2[0][0])


		b1 = -m1*t1[1][1] +t1[0][1]
		b2 = -m2*t2[0][1] + t2[1][1]

		intersectCenter = np.array([(m1*b2+b1)/(1.-m1*m2),m2*((m1*b2+b1)/(1.-m1*m2))+b2])
		# q4 = intersectCenter
		# q4 = [q4[1],q4[0]]
		vertVector = np.array([(t1[0][1]-t1[0][0]),(t1[1][1]-t1[1][0])])
		vertVector = vertVector/np.linalg.norm(vertVector)

		if vertVector[1]<0:
			vertVector = -vertVector

		horizVector = np.array([(t2[0][1]-t2[0][0]),(t2[1][1]-t2[1][0])])
		horizVector = horizVector/np.linalg.norm(horizVector)
		if horizVector[0]<0:
			horizVector = -horizVector

		tH1 = 180./np.pi*np.arctan2(horizVector[1],horizVector[0])
		tV1 =180./np.pi*np.arctan2(vertVector[1],vertVector[0])
		tV1 = tV1 % 360.
		tH1 = tH1 if abs(tH1) < (90.) else tH1 -180.
		angVert = tV1
		angHoriz = tH1
		# angVert =np.arctan2((t1[1][1]-t1[1][0]),(t1[0][1]-t1[0][0]))
		# angHoriz = np.arctan2((t2[1][1]-t2[1][0]),(t2[0][1]-t2[0][0]))
		# angVert = (180*angVert/np.pi) % 360.
		# angHoriz = 180.*angHoriz/np.pi if abs(angHoriz) < (np.pi/2) else 180.*angHoriz/np.pi -180.
		if i1==0:
			edge = Circle(q4,radius=(halfWidth+halfHeight)/2+3,color='black',fill=False)
			ax.add_patch(edge)
			if quadList[l] ==1:
				topRightQuad = Wedge(intersectCenter,(halfWidth+halfHeight)/2+3,angHoriz,angVert,alpha =0.3,color = 'orange')
				bottomLeftQuad = Wedge(intersectCenter,(halfWidth+halfHeight)/2+3,180+angHoriz,180+angVert,alpha =0.3,color = 'orange')
				ax.add_patch(topRightQuad)
				ax.add_patch(bottomLeftQuad)
			elif quadList[l] ==2:
				topLeftQuad = Wedge(intersectCenter,(halfWidth+halfHeight)/2+3,angVert,180+angHoriz,alpha =0.3,color = 'orange')
				bottomRightQuad = Wedge(intersectCenter,(halfWidth+halfHeight)/2+3,180+angVert,360+angHoriz,alpha =0.3,color = 'orange')
				ax.add_patch(topLeftQuad)
				ax.add_patch(bottomRightQuad)

	ax.set_xlim([q4[0]-halfWidth-5,q4[0]+halfWidth+5])
	ax.set_ylim([q4[1]-halfHeight-5,q4[1]+halfHeight+5])
	ax.set_aspect('equal')
	plt.gca().invert_yaxis()
	plt.show()

	while True:
		try:
			move_nums = input("which move?:  ")
			if "," in move_nums:
				move_nums= [int(kk) for kk in move_nums.split(",")]
			else:
				move_nums = [int(move_nums)]
			time_bef = 0
			print(move_nums)
			for move_num in move_nums:
				# print time_bef

				if move_num >0:# and move_num<100:
					# for i1 in xrange(0,move_num-1):
					# 	time_bef+=listFramesInTempB[g1[i1]]
					inThetas.append(angList[g1[move_num-1]])
					outThetas.append(color[g1[move_num-1]])
					decTime.append(time_bef)
					eventNum.append(g1[move_num-1])
					eventFrames.append(list1[g1[move_num-1]])
					eventFNames.append(groupFnames[i-1])
					eventType.append('turn')
				if move_num <0:# and move_num<100:
					# for i1 in xrange(0,move_num-1):
					# 	time_bef+=listFramesInTempB[g1[i1]]
					inThetas.append(angList[g1[move_num]])
					outThetas.append(color[g1[move_num]])
					decTime.append(time_bef)
					eventNum.append(g1[move_num])
					eventFrames.append(list1[g1[move_num]])
					eventFNames.append(groupFnames[i-1])
					eventType.append('turn')

				# elif move_num<0:
				# 	move_num = -move_num
				# 	for i1 in xrange(0,move_num-1):
				# 		time_bef+=listFramesInTempB[g1[i1]]
				# 	inThetas.append(angList[g1[move_num-1]])
				# 	outThetas.append(color[g1[move_num-1]])
				# 	decTime.append(time_bef)
				# 	eventNum.append(g1[move_num-1])
				# 	eventType.append('crossing')

				# elif move_num>100:
				# 	move_num = move_num -100
				# 	for i1 in xrange(0,move_num-1):
				# 		time_bef+=listFramesInTempB[g1[i1]]
				# 	inThetas.append(angList[g1[move_num-1]])
				# 	outThetas.append(color[g1[move_num-1]])
				# 	decTime.append(time_bef)
				# 	eventNum.append(g1[move_num-1])
				# 	eventType.append('stop')
		except:
			print ("Not a possible value. Try again.")
			continue
		break
	print (len(groups)-i, " left!")
	i+=1


responseData = (inThetas,outThetas,decTime,eventNum,eventType, l,listFNames,eventFrames,eventFNames) 
outFile = inputDir.split("/")[-2] + ".response"
pickle.dump(responseData,open(outFile,"wb"))

