import os

# runs original group stuff
groups=['Kir2021']
if 0:
    for groupName in groups:
        os.system("python run_group.py videos/"+groupName)
        os.system("python basic_stats_track.py "+groupName)
        os.system("python basic_boxplots.py "+groupName)
        os.system("python run_all_plots.py "+groupName)


# runs staff plot flex and bdry analysis flex
#groups=['61933FL50', '61933Kir', 'KirFL50', 'FL50']
groups=['FL50', '61933Kir']
if 1: 
    temps = ['30', '35', '40']
    for groupName in groups:
        for t in temps:
            print('Running staff analysis for '+groupName+' at temp '+t)
            os.system('python staff_plot_flex.py outputs/outputs_'+groupName+'/output_'+t+'/')
            print('Running boundary analysis for '+groupName+' at temp '+t)
            os.system('python boundary_analysis_flex.py outputs/outputs_'+groupName+'/output_'+t+'/')
        print('Running turn cross plots for '+groupName)
        os.system('python turn_cross_plots.py '+groupName)