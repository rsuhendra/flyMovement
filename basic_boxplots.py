import pickle
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = "Arial"


groupName = str(sys.argv[1])
bigdir='basic_stats/stats_'+groupName
f1 = bigdir+'/output_40_'+groupName+'_basic_stats.pkl'
r1 = open(f1,'rb')
(allHotFracs40,allAvSpeeds40,all_distTravs40,allColdSpeeds40,allHotSpeeds40,allFnames40)  =pickle.load(r1)
r1.close()
f1 = bigdir+'/output_35_'+groupName+'_basic_stats.pkl'
r1 = open(f1,'rb')
(allHotFracs35,allAvSpeeds35,all_distTravs35,allColdSpeeds35,allHotSpeeds35,allFnames35)  =pickle.load(r1)
r1.close()
f1 = bigdir+'/output_30_'+groupName+'_basic_stats.pkl'
r1 = open(f1,'rb')
(allHotFracs30,allAvSpeeds30,all_distTravs30,allColdSpeeds30,allHotSpeeds30,allFnames30) =pickle.load(r1)
r1.close()
f1 = bigdir+'/output_25_'+groupName+'_basic_stats.pkl'
r1 = open(f1,'rb')
(allHotFracs25,allAvSpeeds25,all_distTravs25,allColdSpeeds25,allHotSpeeds25,allFnames25) =pickle.load(r1)
r1.close()

cold = 0
if cold:
	f1 = bigdir+'/output_20_'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs20,allAvSpeeds20,all_distTravs20,allColdSpeeds20,allHotSpeeds20,allFnames20) =pickle.load(r1)
	r1.close()

	f1 = bigdir+'/output_15_'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs15,allAvSpeeds15,all_distTravs15,allColdSpeeds15,allHotSpeeds15,allFnames15) =pickle.load(r1)
	r1.close()

# building dataframe of all data from our files. Will eventually get stored in a csv for later access. 
df = pd.DataFrame(columns=['hotFrac','av_speed','distance','temperature','cold_speed','hot_speed'])
for i1 in range(len(allHotFracs40)):
	df= df.append({'hotFrac': 1.-2.*allHotFracs40[i1],'av_speed': allAvSpeeds40[i1],'distance':all_distTravs40[i1],'cold_speed': allColdSpeeds40[i1], 'hot_speed':allHotSpeeds40[i1],'temperature':40,'fname':allFnames40[i1]},ignore_index=True)

for i1 in range(len(allHotFracs35)):
	df= df.append({'hotFrac': 1.-2.*allHotFracs35[i1],'av_speed': allAvSpeeds35[i1],'distance':all_distTravs35[i1],'cold_speed': allColdSpeeds35[i1], 'hot_speed':allHotSpeeds35[i1],'temperature':35,'fname':allFnames35[i1]},ignore_index=True)

for i1 in range(len(allHotFracs30)):
	df= df.append({'hotFrac': 1-2.*allHotFracs30[i1],'av_speed': allAvSpeeds30[i1],'distance':all_distTravs30[i1],'cold_speed': allColdSpeeds30[i1], 'hot_speed':allHotSpeeds30[i1],'temperature':30,'fname':allFnames30[i1]},ignore_index=True)

for i1 in range(len(allHotFracs25)):
	df= df.append({'hotFrac':1.-2.*allHotFracs25[i1],'av_speed': allAvSpeeds25[i1],'distance':all_distTravs25[i1],'cold_speed': allColdSpeeds25[i1], 'hot_speed':allHotSpeeds25[i1],'temperature':25,'fname':allFnames25[i1]},ignore_index=True)

if cold:
	for i1 in range(len(allHotFracs20)):
		df= df.append({'hotFrac': 1.-2.*allHotFracs20[i1],'av_speed': allAvSpeeds20[i1],'distance':all_distTravs20[i1],'cold_speed': allColdSpeeds20[i1], 'hot_speed':allHotSpeeds20[i1],'temperature':20,'fname':allFnames20[i1]},ignore_index=True)

	for i1 in range(len(allHotFracs15)):
		df= df.append({'hotFrac':1.-2.*allHotFracs15[i1],'av_speed': allAvSpeeds15[i1],'distance':all_distTravs15[i1],'cold_speed': allColdSpeeds15[i1], 'hot_speed':allHotSpeeds15[i1],'temperature':15,'fname':allFnames15[i1]},ignore_index=True)


if not os.path.exists('./basic_plots/'):
	os.makedirs('./basic_plots/')
if not os.path.exists('./basic_plots/boxplots_'+groupName+'/'):
	os.makedirs('./basic_plots/boxplots_'+groupName+'/')


import seaborn as sns
df['hotFrac'] = df['hotFrac'].astype(float)
df['av_speed'] = df['av_speed'].astype(float)
df['distance'] = df['distance'].astype(float)

print ('Average speed:', np.mean(df['av_speed']),np.std(df['av_speed']))

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='hotFrac',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='hotFrac',data = df,color=".25")
ax.set_ylabel('Avoidance')
ax.set_ylim([-1,1])
plt.savefig('basic_plots/boxplots_'+groupName+'/'+groupName+'_box_avoidance.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='av_speed',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='av_speed',data = df,color=".25")
ax.set_ylabel('Average speed while moving (mm/s)')
ax.set_ylim([0,14])
plt.savefig('basic_plots/boxplots_'+groupName+'/'+groupName+'_box_avSpeed.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='distance',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='distance',data = df,color=".25")
ax.set_ylabel('Distance traveled (mm)')
ax.set_ylim([0,500])
plt.savefig('basic_plots/boxplots_'+groupName+'/'+groupName+'_box_distance.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='hot_speed',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='hot_speed',data = df,color=".25")
ax.set_ylabel('Average Hot Speed (mm/s)')
ax.set_ylim([0,14])
plt.savefig('basic_plots/boxplots_'+groupName+'/'+groupName+'_box_hot_speed.pdf')
plt.close()


fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='cold_speed',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='cold_speed',data = df,color=".25")
ax.set_ylabel('Average Cold Speed (mm/s)')
ax.set_ylim([0,14])
plt.savefig('basic_plots/boxplots_'+groupName+'/'+groupName+'_box_cold_speed.pdf')
plt.close()

df.to_csv('basic_plots/boxplots_'+groupName+'/'+groupName+'_boxdata.csv')
