import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = "Arial"

groupName = str(sys.argv[1])#'A1Kir'


f1 = 'staff/staff_data/staff_data_' + groupName + '/output_40_boxplot_data.pkl'
r1 = open(f1,'rb')
(turnsInVids40,straightsInVids40,crossInsInVids40,paradoxTurnsInVids40,allFnames40)  = pickle.load(r1)
r1.close()

f1 = 'staff/staff_data/staff_data_' + groupName + '/output_35_boxplot_data.pkl'
r1 = open(f1,'rb')
(turnsInVids35,straightsInVids35,crossInsInVids35,paradoxTurnsInVids35,allFnames35)  =pickle.load(r1)
r1.close()

f1 = 'staff/staff_data/staff_data_' + groupName + '/output_30_boxplot_data.pkl'
r1 = open(f1,'rb')
(turnsInVids30,straightsInVids30,crossInsInVids30,paradoxTurnsInVids30,allFnames30)  =pickle.load(r1)
r1.close()


allFnames40 = list(set(allFnames40))
allFnames35 = list(set(allFnames35))
allFnames30 = list(set(allFnames30))

# trimmed = pickle.load(open('trimList_HC_Kir.pkl','rb'))
# trimmed = set([str(t00) for t00 in trimmed])
# building dataframe of all data from our files. Will eventually get stored in a csv for later access. 
df = pd.DataFrame(columns=['turns','straights','cross-ins','paradoxTurns','filename','temperature'])
for fn1 in allFnames40:
	if straightsInVids40[fn1]>5:
		print (fn1)

	# #### quick removal of offending Kir/+, HC/+ flies. 
	# if fn1[0:len(fn1)-9] in trimmed:
	# 	continue
	# else:
	df= df.append({'turns': turnsInVids40[fn1],'straights':straightsInVids40[fn1],'cross-ins':crossInsInVids40[fn1],'paradoxTurns': paradoxTurnsInVids40[fn1],'filename':fn1,'temperature':40},ignore_index=True)


for fn1 in allFnames35:
	df = df.append({'turns': turnsInVids35[fn1],'straights':straightsInVids35[fn1],'cross-ins':crossInsInVids35[fn1],'paradoxTurns': paradoxTurnsInVids35[fn1],'filename':fn1,'temperature':35},ignore_index=True)


for fn1 in allFnames30:
	df = df.append({'turns': turnsInVids30[fn1],'straights':straightsInVids30[fn1],'cross-ins':crossInsInVids30[fn1],'paradoxTurns': paradoxTurnsInVids30[fn1],'filename':fn1,'temperature':30},ignore_index=True)


import seaborn as sns
df['turns'] = df['turns'].astype(float)
df['straights'] = df['straights'].astype(float)
df['cross-ins'] = df['cross-ins'].astype(float)
df['paradoxTurns'] = df['paradoxTurns'].astype(float)


fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='turns',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='turns',data = df,color=".25")
ax.set_ylim([0,16])
plt.savefig('staff/staff_plots/staff_plots_' + groupName + '/'+groupName+'box_turns.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='straights',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='straights',data = df,color=".25")
ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_' + groupName + '/'+groupName+'box_straights.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='cross-ins',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='cross-ins',data = df,color=".25")
ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_' + groupName + '/'+groupName+'box_crossIns.pdf')
plt.close()

fig,ax = plt.subplots()
ax = sns.boxplot(x='temperature',y='paradoxTurns',data = df,fliersize=0.)
ax = sns.swarmplot(x='temperature',y='paradoxTurns',data = df,color=".25")
ax.set_ylim([0,12])
plt.savefig('staff/staff_plots/staff_plots_' + groupName + '/'+groupName+'box_paradoxTurns.pdf')
plt.close()

df.to_csv('staff/staff_data/staff_data_' + groupName + '/boxplot'+groupName+'_data.csv',sep=',')

dat30 = df[df.temperature ==30]
ratio30  = dat30['turns'].sum()/(dat30['turns'].sum() + dat30['straights'].sum())
dat35 = df[df.temperature ==35]
ratio35  = dat35['turns'].sum()/(dat35['turns'].sum() + dat35['straights'].sum())
dat40 = df[df.temperature ==40]
ratio40  = dat40['turns'].sum()/(dat40['turns'].sum() + dat40['straights'].sum())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

print('30',dat30['turns'].sum(),dat30['straights'].sum())
print('35',dat35['turns'].sum(),dat35['straights'].sum())
print('40',dat40['turns'].sum(),dat40['straights'].sum())

fig,ax = plt.subplots()
width =0.8
tRatios = [ratio30,ratio35,ratio40]
ax.bar([1,2,3],tRatios,width,color=['yellow','orange','red'])
ax.bar([1,2,3],[1-ratio30,1-ratio35,1-ratio40],width,bottom=tRatios,color='grey')
ax.set_aspect(3)
plt.savefig('staff/staff_plots/staff_plots_' + groupName + '/'+groupName+'box_ratio_turns.pdf')


print ('Turns at 30: ', np.mean(dat30['turns']) ,'+/-' , np.std(dat30['turns']))
print ('Turns at 35: ',np.mean(dat35['turns']) ,'+/-' , np.std(dat35['turns']))
print ('Turns at 40: ',np.mean(dat40['turns']) ,'+/-', np.std(dat40['turns']))

print ('straights at 30: ', np.mean(dat30['straights']) ,'+/-' , np.std(dat30['straights']))
print ('straights at 35: ',np.mean(dat35['straights']) ,'+/-' , np.std(dat35['straights']))
print ('straights at 40: ',np.mean(dat40['straights']) ,'+/-', np.std(dat40['straights']))



print ('Median Turns at 30: ', np.median(dat30['turns']) ,'Max Turns 30: ' , np.max(dat30['turns']))
print ('Median Turns at 35: ',np.median(dat35['turns']) ,'Max Turns 35: ' , np.max(dat35['turns']))
print ('Median Turns at 40: ',np.median(dat40['turns']) ,'Max Turns 35: ', np.max(dat40['turns']))


print ('Median straights at 30: ', np.median(dat30['straights']) ,'Max straights 30: ' , np.max(dat30['straights']))
print ('Median straights at 35: ',np.median(dat35['straights']) ,'Max straights 35: ' , np.max(dat35['straights']))
print ('Median straights at 40: ',np.median(dat40['straights']) ,'Max straights 40: ', np.max(dat40['straights']))
