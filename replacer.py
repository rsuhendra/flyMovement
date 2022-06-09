fin = open("test2.py", "rt")
#output file to write the result to
fout = open("staff_plot_flex.py", "wt")
#for each line in the input file
for line in fin:
	#read replace the string and write to output file
	fout.write(line.replace("'staff/staff_plots/staff_plots_'+groupName+'/staff_'+tempName+'/'", "'staff/staff_plots/staff_plots_'+groupName+'/'"))
#close input and output files
fin.close()
fout.close()

#plt.savefig('staff/staff_'+groupName+'/staff_'+tempName+'/plots/'+inputDir.split('/')[-2] + '_every_response.svg')
#plt.savefig('staff/staff_plots/staff_plots_'+groupName+'/staff_'+tempName+'/'+inputDir.split('/')[-2] + '_every_response.svg')