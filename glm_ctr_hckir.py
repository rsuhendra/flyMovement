import pickle
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pymer4.models import Lmer

## repeated measures variation of ratio test. 
# groupName ='LRablated'# 'HCKir'# 
groupName ='61933Kir'# 'HCKir'# 
temps = [30,35,40]
print('Working on ', groupName)
for t1 in temps:
	t = str(t1)
	print('Testing for temp ', t)
	f1 = 'bdryflex/bdryflex_data/bdryflex_data_'+groupName+'/output_'+t+'_byVideo_ratio_data.pkl'
	r1 = open(f1,'rb')
	interactionData40_mut  =pickle.load(r1)
	r1.close()

	controlName = 'KirFL50'
	f1 = 'bdryflex/bdryflex_data/bdryflex_data_'+controlName+'/output_'+t+'_byVideo_ratio_data.pkl'
	r1 = open(f1,'rb')
	interactionData40_c  =pickle.load(r1)
	r1.close()

	controlName = '61933FL50'
	f1 = 'bdryflex/bdryflex_data/bdryflex_data_'+controlName+'/output_'+t+'_byVideo_ratio_data.pkl'
	r1 = open(f1,'rb')
	interactionData40_c2  =pickle.load(r1)
	r1.close() 

	controlName = 'FL50'
	f1 = 'bdryflex/bdryflex_data/bdryflex_data_'+controlName+'/output_'+t+'_byVideo_ratio_data.pkl'
	r1 = open(f1,'rb')
	interactionData40_c3  =pickle.load(r1)
	r1.close()


	### put everything into a dataframe

	df = pd.DataFrame(columns=['conditionHC','fly','turnOrNot','time','conditionKir'])

	for i in range(0,len(interactionData40_c)):
		df = df.append({'conditionHC':0,'conditionKir':1,'fly':interactionData40_c[i][0][0:len(interactionData40_c[i][0])-9],'turnOrNot':interactionData40_c[i][1],'time':interactionData40_c[i][2]/30.},ignore_index=True)
	for i in range(0,len(interactionData40_c2)):
		df = df.append({'conditionHC':1,'conditionKir':0,'fly':interactionData40_c2[i][0][0:len(interactionData40_c2[i][0])-9],'turnOrNot':interactionData40_c2[i][1],'time':interactionData40_c2[i][2]/30.},ignore_index=True)
	for i in range(0,len(interactionData40_c3)):
		df = df.append({'conditionHC':0,'conditionKir':0,'fly':interactionData40_c3[i][0][0:len(interactionData40_c3[i][0])-9],'turnOrNot':interactionData40_c3[i][1],'time':interactionData40_c3[i][2]/30.},ignore_index=True)
	for i in range(0,len(interactionData40_mut)):
		df = df.append({'conditionHC':1,'conditionKir':1,'fly':interactionData40_mut[i][0][0:len(interactionData40_mut[i][0])-9],'turnOrNot':interactionData40_mut[i][1],'time':interactionData40_mut[i][2]/30.},ignore_index=True)
	# print(df.shape)

	# if t1==40:
	# 	#### remove kir/+ flies that just walk straight at 40. 
	# 	a1 = df[df['conditionKir']==1]
	# 	a1 = a1[a1['conditionHC']==0]
	# 	trimList1 = [a00 for a00 in np.unique(a1['fly']) if a1[a1['fly']==a00]['turnOrNot'].mean()<=0.7 and a1[a1['fly']==a00]['turnOrNot'].shape[0]>=2]
	# 	# print(a1.shape)
	# 	a1 = a1[~a1['fly'].isin(trimList1)]
	# 	# print(a1.shape)
	# 	a0 = a1.copy()
	# 	df = df[~df['fly'].isin(trimList1)]
	# 	# print(df.shape)
	# 	# print(sum(~df['fly'].isin(trimList1)))
	# 	# asdf
	# 	a1 = df[df['conditionKir']==0]
	# 	a1 = a1[a1['conditionHC']==1]
	# 	a01 = a1.copy()
	# 	trimList = [a00 for a00 in np.unique(a1['fly']) if a1[a1['fly']==a00]['turnOrNot'].mean()<=0.6 and a1[a1['fly']==a00]['turnOrNot'].shape[0]>=2]
	# 	df = df[~df['fly'].isin(trimList)]

	# 	pickle.dump(trimList1+trimList, open('trimList_HC_Kir.pkl','wb'),protocol=2)

	# 	a1 = df[df['conditionKir']==1]
	# 	a1 = a1[a1['conditionHC']==1]

	df['fly'] = df['fly'].astype('category')
	df['time'] = df['time'].astype(float)
	df['turnOrNot'] = df['turnOrNot'].astype(int)
	#df.to_csv(t+'.csv',index=False)

	'''
	df['conditionHC'] = df['conditionHC'].astype(bool)
	df['conditionKir'] = df['conditionKir'].astype(bool)
	df['turnOrNot'] = df['turnOrNot'].astype(bool)
	'''
	
	kirTC = df[(df['conditionHC']==0) & (df['conditionKir']==1)]
	hcTC = df[(df['conditionHC']==1) & (df['conditionKir']==0)]
	hcKirTC = df[(df['conditionHC']==1) & (df['conditionKir']==1)]
	wild = df[(df['conditionHC']==0) & (df['conditionKir']==0)]
	print('wild',wild['turnOrNot'].sum(),len(wild['turnOrNot']),wild['turnOrNot'].sum()/len(wild['turnOrNot']))
	print('kir+',kirTC['turnOrNot'].sum(),len(kirTC['turnOrNot']),kirTC['turnOrNot'].sum()/len(kirTC['turnOrNot']))
	print('hc+',hcTC['turnOrNot'].sum(),len(hcTC['turnOrNot']),hcTC['turnOrNot'].sum()/len(hcTC['turnOrNot']))
	print('hckir',hcKirTC['turnOrNot'].sum(),len(hcKirTC['turnOrNot']),hcKirTC['turnOrNot'].sum()/len(hcKirTC['turnOrNot']))

	#or can do as condition*time
	model = Lmer('turnOrNot ~ conditionHC*conditionKir + (1|fly)', data=df, family='binomial')
	model.fit() #conf_int='boot',n_boot=1000)
	print(model.coefs)

	# print(model.anova(force_orthon))
