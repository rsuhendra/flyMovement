
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *
import os
import StringIO
from PIL import Image


# dataFname = "./output/"
inputF = str(sys.argv[1])

(inThetas,outThetas,decTime,eventNum,l,listFNames) = pickle.load(open(inputF,"rb"))


fig,ax=plt.subplots()
scat2 =ax.scatter(inThetas,outThetas,c=np.log(np.array(decTime)+1),picker=5,s=10)
plt.colorbar(scat2,ax=ax)
ax.set_xlabel('initial angle relative to temp wall')
ax.set_ylabel('resulting turn angle')
ax.set_xlim([0,180])
# fig.canvas.mpl_connect('pick_event', onpick_temp_boundary)
plt.show()

groupSmall = []
groupLarge = []
for i in xrange(0, len(inThetas)):
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
plt.savefig('allTurnsHist.png')


from sklearn.neighbors.kde import KernelDensity
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

plt.savefig('smoothedVals_selected_'+inputF.split('.')[0]+'.png')
import json
class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)
jOut = [color]
jDat = json.dumps(jOut,cls=NumpyEncoder)
with open(inputF.split('.')[0]+'_turningVals.json','w') as f:
	json.dump(jDat,f)

bins = np.linspace(0,180,6)
inds = np.digitize(inThetas,bins,right=True)
print inds,bins
numSlots = len(bins)
rCount = np.zeros(numSlots)
lCount = np.zeros(numSlots)
sCount = np.zeros(numSlots)
angThres = 10
for i in xrange(0,len(inThetas)):
	if outThetas[i]>angThres:
		rCount[inds[i]-1]+=1

	elif outThetas[i]<-angThres:
		lCount[inds[i]-1]+=1

	else:
		sCount[inds[i]-1]+=1

for i in xrange(0,len(rCount)):
	tot = rCount[i]+lCount[i]+sCount[i]
	rCount[i] /=tot
	sCount[i]/=tot
	lCount[i] /= tot
fig, ax = plt.subplots()
plt.stackplot(bins,[rCount,sCount,lCount],labels=['Right','Straight','Left'])
plt.legend(loc='upper left')
plt.savefig('prob_turn_'+inputF.split('.')[0]+'.png')