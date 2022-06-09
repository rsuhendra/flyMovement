import pickle
import numpy as np
import sys

import matplotlib.pyplot as plt
import pandas as pd


def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov


# building dataframe of all data from our files. Will eventually get stored in a csv for later access. 
df = pd.DataFrame(columns=['gname','hotFrac','av_speed','distance','temperature','cold_speed','hot_speed'])

controlName = 'FL50combo'
groupNames = ['LRablated']
allGroupNames = groupNames + [controlName]
for groupName in allGroupNames:
	f1 = 'output_40'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs40,allAvSpeeds40,all_distTravs40,allColdSpeeds40,allHotSpeeds40,allFnames40)  =pickle.load(r1)
	r1.close()
	
	f1 = 'output_35'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs35,allAvSpeeds35,all_distTravs35,allColdSpeeds35,allHotSpeeds35,allFnames35)  =pickle.load(r1)
	r1.close()

	f1 = 'output_30'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs30,allAvSpeeds30,all_distTravs30,allColdSpeeds30,allHotSpeeds30,allFnames30) =pickle.load(r1)
	r1.close()

	f1 = 'output_25'+groupName+'_basic_stats.pkl'
	r1 = open(f1,'rb')
	(allHotFracs25,allAvSpeeds25,all_distTravs25,allColdSpeeds25,allHotSpeeds25,allFnames25) =pickle.load(r1)
	r1.close()


	cold =1
	if cold:
		f1 = 'output_20'+groupName+'_basic_stats.pkl'
		r1 = open(f1,'rb')
		(allHotFracs20,allAvSpeeds20,all_distTravs20,allColdSpeeds20,allHotSpeeds20,allFnames20) =pickle.load(r1)
		r1.close()

		f1 = 'output_15'+groupName+'_basic_stats.pkl'
		r1 = open(f1,'rb')
		(allHotFracs15,allAvSpeeds15,all_distTravs15,allColdSpeeds15,allHotSpeeds15,allFnames15) =pickle.load(r1)
		r1.close()

	for i1 in xrange(0,len(allHotFracs40)):
		df= df.append({'gname':groupName,'hotFrac': 1-2*allHotFracs40[i1],'av_speed': allAvSpeeds40[i1],'distance':all_distTravs40[i1],'cold_speed': allColdSpeeds40[i1], 'hot_speed':allHotSpeeds40[i1],'temperature':40},ignore_index=True)

	for i1 in xrange(0,len(allHotFracs35)):
		df= df.append({'gname':groupName,'hotFrac': 1-2.*allHotFracs35[i1],'av_speed': allAvSpeeds35[i1],'distance':all_distTravs35[i1],'cold_speed': allColdSpeeds35[i1], 'hot_speed':allHotSpeeds35[i1],'temperature':35},ignore_index=True)

	for i1 in xrange(0,len(allHotFracs30)):
		df= df.append({'gname':groupName,'hotFrac': 1-2.*allHotFracs30[i1],'av_speed': allAvSpeeds30[i1],'distance':all_distTravs30[i1],'cold_speed': allColdSpeeds30[i1], 'hot_speed':allHotSpeeds30[i1],'temperature':30},ignore_index=True)

	for i1 in xrange(0,len(allHotFracs25)):
		df= df.append({'gname':groupName,'hotFrac':1-2.*allHotFracs25[i1],'av_speed': allAvSpeeds25[i1],'distance':all_distTravs25[i1],'cold_speed': allColdSpeeds25[i1], 'hot_speed':allHotSpeeds25[i1],'temperature':25},ignore_index=True)

	if cold:
		for i1 in xrange(0,len(allHotFracs20)):
			df= df.append({'gname':groupName,'hotFrac': 1-2.*allHotFracs20[i1],'av_speed': allAvSpeeds20[i1],'distance':all_distTravs20[i1],'cold_speed': allColdSpeeds20[i1], 'hot_speed':allHotSpeeds20[i1],'temperature':20},ignore_index=True)

		for i1 in xrange(0,len(allHotFracs15)):
			df= df.append({'gname':groupName,'hotFrac':1-2.*allHotFracs15[i1],'av_speed': allAvSpeeds15[i1],'distance':all_distTravs15[i1],'cold_speed': allColdSpeeds15[i1], 'hot_speed':allHotSpeeds15[i1],'temperature':15},ignore_index=True)


import seaborn as sns
# df['hotFrac'] = df['hotFrac'].astype(float)
# df['av_speed'] = df['av_speed'].astype(float)
# df['distance'] = df['distance'].astype(float)
from scipy.stats import mannwhitneyu,ttest_ind,f_oneway,levene,anderson,shapiro,kstest,kruskal,ttest_1samp
### do comparisons of hot speed
import csv
import statsmodels.api as sm
from statsmodels.formula.api import ols
temps = [30,35,40]
if cold:
	temps = [15,20,25]+temps
with open('speed_avoid_stats_'+groupNames[0]+'_'+controlName+'.csv',mode='w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['test_hot_speed'])
	writer.writerow(['control','comparison','temperature','stats','df'])

	for t1 in temps:
		control1 = df['hot_speed'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['hot_speed']))]
		# print control1
		for g1 in groupNames:
			test1 = df['hot_speed'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['hot_speed']))]
			writer.writerow([controlName,g1,t1,ttest_ind(control1,test1,equal_var=False)])#switch to anova.... 
			writer.writerow([controlName,g1,t1,f_oneway(control1,test1),len(control1)+len(test1)-1])#switch to anova.... 
			# writer.writerow([controlName,g1,t1,kruskal(control1,test1)]) 
			writer.writerow([controlName,g1,t1,kstest(control1,'norm',args=(np.mean(control1),np.std(control1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,kstest(test1,'norm',args=(np.mean(test1),np.std(test1)))])#switch to anova....


			#### build small df for running anova. 
			df0 = pd.DataFrame(columns=['hot_speed','test'])
			for c1 in control1:
				df0 = df0.append({'hot_speed':c1,'test':False},ignore_index=True)
			for t1 in test1:
				df0 = df0.append({'hot_speed':t1,'test':True},ignore_index=True)
			results = ols('hot_speed ~ C(test)', data=df0).fit()
			table = sm.stats.anova_lm(results, typ=2)

			table = anova_table(table)
			writer.writerow([controlName,g1,'estimate','fstat','pval','CI','omega_sq'])
			writer.writerow([controlName,g1,results.params[1],table['F'][0],table['PR(>F)'][0],list(results.conf_int().iloc[1,:]),table['omega_sq'][0]])
# comparisons of cold speed
	writer.writerow(['test_cold_speed'])
	writer.writerow(['control','comparison','temperature','stats','df'])

	for t1 in temps:
		control1 = df['cold_speed'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['cold_speed']))]
		# print control1
		for g1 in groupNames:
			test1 = df['cold_speed'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['cold_speed']))]
			writer.writerow([controlName,g1,t1,ttest_ind(control1,test1,equal_var=False)])
			writer.writerow([controlName,g1,t1,f_oneway(control1,test1),len(control1)+len(test1)-1])#switch to anova....
			# writer.writerow([controlName,g1,t1,kruskal(control1,test1)])  
			writer.writerow([controlName,g1,t1,kstest(control1,'norm',args=(np.mean(control1),np.std(control1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,kstest(test1,'norm',args=(np.mean(test1),np.std(test1)))])#switch to anova....
# comparisons of avoidance

	writer.writerow(['test_avoidance'])
	writer.writerow(['control','comparison','temperature','stats'])

	for t1 in temps:
		control1 = df['hotFrac'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['hotFrac']))]
		print np.mean(control1),'hi'
		for g1 in groupNames:
			test1 = df['hotFrac'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['hotFrac']))]
			print np.mean(test1)
			writer.writerow([controlName,g1,t1,ttest_ind(control1,test1,equal_var=False)])
			writer.writerow([controlName,g1,t1,f_oneway(control1,test1),len(control1)+len(test1)-1])#switch to anova....
			# writer.writerow([controlName,g1,t1,kruskal(control1,test1)]) 
			writer.writerow([controlName,g1,t1,kstest(control1,'norm',args=(np.mean(control1),np.std(control1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,kstest(test1,'norm',args=(np.mean(test1),np.std(test1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,ttest_1samp(test1,0)])#switch to anova....


			#### build small df for running anova. 
			df0 = pd.DataFrame(columns=['hotFrac','test'])
			for c1 in control1:
				df0 = df0.append({'hotFrac':c1,'test':False},ignore_index=True)
			for t1 in test1:
				df0 = df0.append({'hotFrac':t1,'test':True},ignore_index=True)
			results = ols('hotFrac ~ C(test)', data=df0).fit()
			table = sm.stats.anova_lm(results, typ=2)

			table = anova_table(table)
			writer.writerow([controlName,g1,'estimate','fstat','pval','CI','omega_sq','Nsamples:',len(test1),len(control1)])
			writer.writerow([controlName,g1,results.params[1],table['F'][0],table['PR(>F)'][0],list(results.conf_int().iloc[1,:]),table['omega_sq'][0]])#switch to anova....
	# #special check 
	# test1 = df['hotFrac'][(df['temperature']==40) & (df['gname']==g1)& (1-np.isnan(df['hotFrac']))]
	# control1 = df['hotFrac'][(df['temperature']==30) & (df['gname']==g1)& (1-np.isnan(df['hotFrac']))]
	# print([controlName,g1,t1,ttest_ind(control1,test1)])

	# writer.writerow(['test_avoidance_variance'])
	# writer.writerow(['control','comparison','temperature','stats'])

	# for t1 in temps:
	# 	control1 = df['hotFrac'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['hotFrac']))]
	# 	print np.mean(control1),'hi'
	# 	for g1 in groupNames:
	# 		test1 = df['hotFrac'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['hotFrac']))]
	# 		print np.mean(test1)
	# 		writer.writerow([controlName,g1,t1,levene(control1,test1)])
	# 		# writer.writerow([controlName,g1,t1,f_oneway(control1,test1)])#switch to anova.... 


	writer.writerow(['test_avg_speed'])
	writer.writerow(['control','comparison','temperature','stats','df'])

	for t1 in temps:
		control1 = df['av_speed'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['av_speed']))]
		print np.mean(control1),'hi'
		for g1 in groupNames:
			test1 = df['av_speed'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['av_speed']))]
			print np.mean(test1)
			writer.writerow([controlName,g1,t1,ttest_ind(control1,test1,equal_var=False)])
			writer.writerow([controlName,g1,t1,f_oneway(control1,test1),len(control1)+len(test1)-1])#switch to anova.... 
			# writer.writerow([controlName,g1,t1,kruskal(control1,test1)]) 
			writer.writerow([controlName,g1,t1,kstest(control1,'norm',args=(np.mean(control1),np.std(control1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,kstest(test1,'norm',args=(np.mean(test1),np.std(test1)))])#switch to anova....

	writer.writerow(['test_distance'])
	writer.writerow(['control','comparison','temperature','stats','df'])

	for t1 in temps:
		control1 = df['distance'][(df['temperature']==t1) & (df['gname']==controlName) & (1-np.isnan(df['distance']))]
		# print np.mean(control1),'hi'
		for g1 in groupNames:
			test1 = df['distance'][(df['temperature']==t1) & (df['gname']==g1)& (1-np.isnan(df['distance']))]
			# print np.mean(test1)
			writer.writerow([controlName,g1,t1,ttest_ind(control1,test1,equal_var=False)])
			writer.writerow([controlName,g1,t1,f_oneway(control1,test1),len(control1)+len(test1)-1])#switch to anova.... 
			# writer.writerow([controlName,g1,t1,kruskal(control1,test1)]) 
			writer.writerow([controlName,g1,t1,kstest(control1,'norm',args=(np.mean(control1),np.std(control1)))])#switch to anova....
			writer.writerow([controlName,g1,t1,kstest(test1,'norm',args=(np.mean(test1),np.std(test1)))])#switch to anova....

if 0:
	###########################
	#testing ratio stuff
	# building dataframe of all data from our files. Will eventually get stored in a csv for later access. 
	df = pd.DataFrame(columns=['gname','turns','straights','cross-ins','paradoxTurns','filename','temperature'])

	for groupName in allGroupNames:
		f1 = 'output_40'+groupName+'_boxplot_data.pkl'
		r1 = open(f1,'rb')
		(turnsInVids40,straightsInVids40,crossInsInVids40,paradoxTurnsInVids40,allFnames40)  =pickle.load(r1)
		r1.close()
		f1 = 'output_35'+groupName+'_boxplot_data.pkl'
		r1 = open(f1,'rb')
		(turnsInVids35,straightsInVids35,crossInsInVids35,paradoxTurnsInVids35,allFnames35)  =pickle.load(r1)
		r1.close()

		f1 = 'output_30'+groupName+'_boxplot_data.pkl'
		r1 = open(f1,'rb')
		(turnsInVids30,straightsInVids30,crossInsInVids30,paradoxTurnsInVids30,allFnames30)  =pickle.load(r1)
		r1.close()


		allFnames40 = list(set(allFnames40))
		allFnames35 = list(set(allFnames35))
		allFnames30 = list(set(allFnames30))

		# building dataframe of all data from our files. Will eventually get stored in a csv for later access. 
		for fn1 in allFnames40:
			if straightsInVids40[fn1]>5:
				print fn1
			df= df.append({'gname':groupName,'turns': turnsInVids40[fn1],'straights':straightsInVids40[fn1],'cross-ins':crossInsInVids40[fn1],'paradoxTurns': paradoxTurnsInVids40[fn1],'filename':fn1,'temperature':40},ignore_index=True)


		for fn1 in allFnames35:
			df = df.append({'gname':groupName,'turns': turnsInVids35[fn1],'straights':straightsInVids35[fn1],'cross-ins':crossInsInVids35[fn1],'paradoxTurns': paradoxTurnsInVids35[fn1],'filename':fn1,'temperature':35},ignore_index=True)


		for fn1 in allFnames30:
			df = df.append({'gname':groupName,'turns': turnsInVids30[fn1],'straights':straightsInVids30[fn1],'cross-ins':crossInsInVids30[fn1],'paradoxTurns': paradoxTurnsInVids30[fn1],'filename':fn1,'temperature':30},ignore_index=True)


	##### testing and writing
	from scipy.stats import fisher_exact
	with open('turn_cross_stats_'+groupNames[0]+'_'+controlName+'.csv',mode='w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['control','comparison','temperature','stats','df'])
		for t1 in temps:
			control1_turns = df['turns'][(df['temperature']==t1) & (df['gname']==controlName) ].tolist()
			control1_straights = df['straights'][(df['temperature']==t1) & (df['gname']==controlName) ].tolist()
			# print control1
			for g1 in groupNames:
				test1_turns = df['turns'][(df['temperature']==t1) & (df['gname']==g1)].tolist()
				test1_straights = df['straights'][(df['temperature']==t1) & (df['gname']==g1)].tolist()
				writer.writerow([controlName,g1,t1,fisher_exact(np.array([[np.sum(control1_turns),np.sum(control1_straights)],[np.sum(test1_turns),np.sum(test1_straights)]])),1])


		# dat30 = df[df.temperature ==30]
		# ratio30  = dat30['turns'].sum()/(dat30['turns'].sum() + dat30['straights'].sum())
		# dat35 = df[df.temperature ==35]
		# ratio35  = dat35['turns'].sum()/(dat35['turns'].sum() + dat35['straights'].sum())
		# dat40 = df[df.temperature ==40]
		# ratio40  = dat40['turns'].sum()/(dat40['turns'].sum() + dat40['straights'].sum())



