# script to run a bunch of things at once.

#from multiprocessing import parent_process
import imageio
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import pickle
from trackImproved_1 import *

# simple script to rerun tracking for all files in a directory (of output files)

temps = ['25','30','35','40']
# temps = ['15','20']+temps

folderWithDataFiles = str(sys.argv[1])
name = folderWithDataFiles.split("/")[-1]
print(name)
for temp in temps:
    outputDir = 'outputs/'+'outputs_'+name+'/'+'output_'+temp+'/'
    CHECK_FOLDER = os.path.isdir(outputDir)
    if not CHECK_FOLDER:
        os.makedirs(outputDir)
        print('Created folder: ' + outputDir)

    for filename in os.listdir(folderWithDataFiles):
        # skip if output file already exists
        if os.path.exists(outputDir+filename.split(".")[0]+'.output'):
            continue
        quad = filename.split("_")[0]
        # print(folderWithDataFiles.split("/")[-2] == 'Kir2021_+', filename.split("_")[1])
        if len(filename.split(".")) < 2:
            print('Filename error')
            continue
        
        if filename.split("_")[1] == '25vs'+temp:
            print('Processing '+filename+' ...')

            # GENOTYPES
            dec2021=['HdBKir','HdB+','DNg06kir','DNg06+','DNB05_Kir','DNB05+','tshGal80_kir',
            'tshGal80_kir_61933','DILP-LexA+','AopKir+','DilpLexA_AopKir']

            jenna2017=['61933FL50', '61933Kir', 'KirFL50']

            early2021 = ['Kir2021']

            summer2022 = ['P6Kir', 'P6+', 'P4_P6_+', 'P4_P6_Kir']

            mbon2017 = ['MBON+', 'MBON_Kir']

            if name in mbon2017:
                tval = 180
                if quad == 'croppedQ1':
                    arenafile = 'mbon17q1.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'mbon17q2.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'mbon17q3.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'mbon17q4.arena'
                else:
                    print('Error!!!!')
                scaling = 4.7
                suppress = 1
                flip = 0

            elif name in dec2021:
                if quad == 'croppedQ1':
                    arenafile = 'q1_prova1_1.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_prova1_1.arena'
                scaling = 4.7
                tval = 150
                suppress = 1
                flip = 1

            elif name in jenna2017:
                if quad == 'croppedQ1' or quad == 'Q1':
                    arenafile = 'q1_jenna3N.arena'
                elif quad == 'croppedQ2' or quad == 'Q2':
                    arenafile = 'q2_jenna3N.arena'
                elif quad == 'croppedQ3' or quad == 'Q3':
                    arenafile = 'q3_jenna3N.arena'
                elif quad == 'croppedQ4' or quad == 'Q4':
                    arenafile = 'q4_jenna3N.arena'
                scaling = 4.7
                tval = 180 
                flip = 0
            
            elif name == 'newFL50':
                if quad == 'croppedQ3' or quad == 'Q3':
                    arenafile = 'q3_10_2018.arena'
                elif quad == 'croppedQ4' or quad == 'Q4':
                    arenafile = 'q4_10_2018.arena'
                scaling = 4.7
                tval = 70
                flip = 0
            
            elif name == 'FL50':
                if quad == 'croppedQ1':
                    arenafile = 'q1_2016.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_2016.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_2016.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_2016.arena'
                scaling = 4.7
                tval = 150
                flip = 0

            elif name == 'merged_FL50':
                scaling = 4.7
                flip = 0
                if filename.split("_")[2].split("-")[2]=="2016":
                    tval = 150
                    if quad == 'croppedQ1':
                        arenafile = 'q1_2016.arena'
                    elif quad == 'croppedQ2':
                        arenafile = 'q2_2016.arena'
                    elif quad == 'croppedQ3':
                        arenafile = 'q3_2016.arena'
                    elif quad == 'croppedQ4':
                        arenafile = 'q4_2016.arena'
                elif filename.split("_")[2].split("-")[2]=="2018":
                    tval = 70
                    if quad == 'croppedQ3' or quad == 'Q3':
                        arenafile = 'q3_10_2018.arena'
                    elif quad == 'croppedQ4' or quad == 'Q4':
                        arenafile = 'q4_10_2018.arena'
                else:
                    print('YOU MESSED UP')

            elif name == 'merged_KirFL50':
                scaling = 4.7
                flip = 0
                #print(filename.split("_")[2].split("-")[2])
                if filename.split("_")[2].split("-")[2] in ["2016", "2017"]:
                    if quad == 'croppedQ1' or quad == 'Q1':
                        arenafile = 'q1_jenna3N.arena'
                    elif quad == 'croppedQ2' or quad == 'Q2':
                        arenafile = 'q2_jenna3N.arena'
                    elif quad == 'croppedQ3' or quad == 'Q3':
                        arenafile = 'q3_jenna3N.arena'
                    elif quad == 'croppedQ4' or quad == 'Q4':
                        arenafile = 'q4_jenna3N.arena'
                    tval = 180 
                elif filename.split("_")[2].split("-")[2]=="2019":
                    if quad == 'croppedQ2':
                        arenafile = 'q2_shtrpa1.arena'
                    elif quad == 'croppedQ4':
                        arenafile = 'q4_shtrpa1.arena'
                    tval = 120
                elif filename.split("_")[2].split("-")[2]=="2021":
                    if float(filename.split("_")[2].split('-')[0]) > 2.:
                        if float(filename.split("_")[2].split('-')[0]) > 3.:
                            tval = 70
                        else:
                            tval = 120
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96midJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96midJan.arena'
                    else:
                        tval = 150
                        if ((float(filename.split("_")[2].split('-')[0])-1)*100 + float(filename.split("_")[2].split('-')[1])) > 22:
                            if quad == 'croppedQ1':
                                arenafile = 'q1_96lateJan.arena'
                            elif quad == 'croppedQ2':
                                arenafile = 'q2_96lateJan.arena'
                        else:
                            if quad == 'croppedQ1':
                                arenafile = 'q1_96midJan.arena'
                            elif quad == 'croppedQ2':
                                arenafile = 'q2_96midJan.arena'
                else:
                    print('YOU MESSED UP BUDDY')

            elif name == 'KirFL50extra':
                if quad == 'croppedQ2':
                    arenafile = 'q2_shtrpa1.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_shtrpa1.arena'
                scaling = 4.7
                tval = 120
                flip = 0

            elif name in early2021:
                print((float(filename.split("_")[2].split('-')[0])-1)*100 + float(filename.split("_")[2].split('-')[1]), 'month-1 + day')
                print(float(filename.split("_")[2].split('-')[0]), 'month')
                if float(filename.split("_")[2].split('-')[0]) > 2.:
                    if float(filename.split("_")[2].split('-')[0]) > 3.:
                        tval = 70
                    else:
                        tval = 120
                    if quad == 'croppedQ1':
                        arenafile = 'q1_96midJan.arena'
                    elif quad == 'croppedQ2':
                        arenafile = 'q2_96midJan.arena'
                else:
                    tval = 150
                    if ((float(filename.split("_")[2].split('-')[0])-1)*100 + float(filename.split("_")[2].split('-')[1])) > 22:
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96lateJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96lateJan.arena'
                    else:
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96midJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96midJan.arena'
                scaling = 4.7
                flip = 0

            elif name in summer2022:
                arenafile = 'q1_05_2022.arena'
                scaling = 4.7
                tval = 200
                flip = 1
            
            else:
                print('YOU MESSED UP BUDDY')

            arenafile = 'arenas/'+arenafile
            suppress = 1
            print("python trackImproved_1.py " + folderWithDataFiles+'/'+filename +' 0 ' + arenafile + ' ' + outputDir + ' '+str(tval) + ' '+str(scaling)+ ' '+str(suppress) + ' '+str(flip))
            os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+filename +' 0 ' + arenafile + ' ' + outputDir + ' '+str(tval) + ' '+str(scaling)+ ' '+str(suppress) + ' '+str(flip))


            """ 
            if folderWithDataFiles.split("/")[-2] == 'FL50' or folderWithDataFiles.split("/")[-2] == 'HC_Kir' or folderWithDataFiles.split("/")[-2] == 'Split22CO6_Kir':
                if quad == 'croppedQ1':
                    arenafile = 'q1_FL50.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_FL50.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_FL50.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_FL50.arena'

            elif folderWithDataFiles.split("/")[-2] == 'Kir2021_+':

                if float(filename.split("_")[2].split('-')[0]) == 4:
                    if quad == 'croppedQ1':
                        arenafile = 'q1_96midJan.arena'
                    elif quad == 'croppedQ2':
                        arenafile = 'q2_96midJan.arena'
                    scaling = 4.7
                    tval = 70
                    continue
                elif float(filename.split("_")[2].split('-')[0]) > 2.:
                    if quad == 'croppedQ1':
                        arenafile = 'q1_96midJan.arena'
                    elif quad == 'croppedQ2':
                        arenafile = 'q2_96midJan.arena'
                    scaling = 4.7
                    tval = 70
                    continue
                else:
                    if float(filename.split("_")[2].split('-')[0]) < 2 and float(filename.split("_")[2].split('-')[1]) > 22:
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96lateJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96lateJan.arena'
                    else:
                        if float(filename.split("_")[2].split('-')[0]) < 2:
                            continue
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96midJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96midJan.arena'

                    scaling = 4.7
                    tval = 150  # 70

                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'PFNdKir' or folderWithDataFiles.split("/")[-2] == 'PFNd+':

                if quad == 'croppedQ1':
                    arenafile = 'q1_96midJan.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_96midJan.arena'
                scaling = 4.7
                tval = 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'PFNvKir' or folderWithDataFiles.split("/")[-2] == 'PFNv+' or folderWithDataFiles.split("/")[-2] == 'PFNvKir2':

                if quad == 'croppedQ1':
                    arenafile = 'q1_96midJan.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_96midJan.arena'
                scaling = 4.7
                tval = 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'R4Kir' or folderWithDataFiles.split("/")[-2] == 'R4+':

                if quad == 'croppedQ1':
                    arenafile = 'q1_96midJan.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_96midJan.arena'
                scaling = 4.7

                if float(filename.split("_")[2].split('-')[0]) < 3:
                    tval = 160
                    continue
                elif float(filename.split("_")[2].split('-')[0]) > 3:
                    tval = 75
                    continue
                else:
                    tval = 30
                    # continue
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'R1Kir' or folderWithDataFiles.split("/")[-2] == 'R1+':

                if quad == 'croppedQ1':
                    arenafile = 'q1_96midJan.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_96midJan.arena'
                scaling = 4.7

                if float(filename.split("_")[2].split('-')[1]) > 15:
                    # asdf
                    tval = 30  # 70
                else:
                    tval = 100

                if float(filename.split("_")[2].split('-')[0]) < 3:
                    tval = 120
                else:
                    continue
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'SS00096_Kir' or folderWithDataFiles.split("/")[-2] == 'Kir2021_+' or folderWithDataFiles.split("/")[-2] == 'SS00096_+':
                print((float(filename.split("_")[2].split(
                    '-')[0])-1)*100 + float(filename.split("_")[2].split('-')[1]), 'check')
                print(float(filename.split("_")[2].split('-')[0]), 'hihi')
                if float(filename.split("_")[2].split('-')[0]) > 2.:
                    if quad == 'croppedQ1':
                        arenafile = 'q1_96midJan.arena'
                    elif quad == 'croppedQ2':
                        arenafile = 'q2_96midJan.arena'
                else:
                    if ((float(filename.split("_")[2].split('-')[0])-1)*100 + float(filename.split("_")[2].split('-')[1])) > 22:
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96lateJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96lateJan.arena'
                    else:
                        if quad == 'croppedQ1':
                            arenafile = 'q1_96midJan.arena'
                        elif quad == 'croppedQ2':
                            arenafile = 'q2_96midJan.arena'
                    continue
                    asdf
                scaling = 4.7
                tval = 120  # 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'DNa2_Kir' or folderWithDataFiles.split("/")[-2] == 'DNa2_+':

                if quad == 'croppedQ2':
                    arenafile = 'q2_DNa2.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_DNa2.arena'

                scaling = 3.9
                tval = 40  # 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'Ablated_L' or folderWithDataFiles.split("/")[-2] == 'Ablated_R' or folderWithDataFiles.split("/")[-2] == 'Ablated_L_R' or folderWithDataFiles.split("/")[-2] == 'MWKir':
                if quad == 'croppedQ1':
                    arenafile = 'q1_LR.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_LR.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_LR.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_LR.arena'

                scaling = 4.7
                tval = 180  # 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == '22C06FL50' or folderWithDataFiles.split("/")[-2] == '22C06Kir' or folderWithDataFiles.split("/")[-2] == 'VT61933Kir' or folderWithDataFiles.split("/")[-2] == 'VT61933FL50' or folderWithDataFiles.split("/")[-2] == 'KirFL50Jenna':
                if quad == 'croppedQ1' or quad == 'Q1':
                    arenafile = 'q1_jenna3N.arena'
                elif quad == 'croppedQ2' or quad == 'Q2':
                    arenafile = 'q2_jenna3N.arena'
                elif quad == 'croppedQ3' or quad == 'Q3':
                    arenafile = 'q3_jenna3N.arena'
                elif quad == 'croppedQ4' or quad == 'Q4':
                    arenafile = 'q4_jenna3N.arena'

                scaling = 4.7
                tval = 180  # 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'Split95C02-19428_Kir' or folderWithDataFiles.split("/")[-2] == 'Split95C02-19428_+':

                scaling = 4.7
                tval = 180  # 70
                if filename.split("_")[1] == '25vs15' or filename.split("_")[1] == '25vs20':
                    scaling = 3.9
                    tval = 100  # 70
                    if quad == 'croppedQ3' or quad == 'Q3':
                        arenafile = 'q3_95c02cold.arena'
                    elif quad == 'croppedQ4' or quad == 'Q4':
                        arenafile = 'q3_95c02cold.arena'
                else:
                    if quad == 'croppedQ1' or quad == 'Q1':
                        arenafile = 'q1_95c02hot.arena'
                    elif quad == 'croppedQ2' or quad == 'Q2':
                        arenafile = 'q2_95c02hot.arena'
                    elif quad == 'croppedQ3' or quad == 'Q3':
                        arenafile = 'q3_95c02hot.arena'
                    elif quad == 'croppedQ4' or quad == 'Q4':
                        arenafile = 'q4_95c02hot.arena'

                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'Dmelanogaster_GR28_8':
                if quad == 'croppedQ1':
                    arenafile = 'q1_Nov2017.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_Nov2017.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_Nov2017.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_Nov2017.arena'
            elif folderWithDataFiles.split("/")[-2] == 'Dmojavensis':
                if quad == 'croppedQ3':
                    arenafile = 'q3_moj.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_moj.arena'

            elif folderWithDataFiles.split("/")[-2] == 'oscillator_Lablated' or folderWithDataFiles.split("/")[-2] == 'oscillator_Rablated' or folderWithDataFiles.split("/")[-2] == 'oscillator_SHAM' or folderWithDataFiles.split("/")[-2] == 'heatRampLablated' or folderWithDataFiles.split("/")[-2] == 'heatRampRablated' or folderWithDataFiles.split("/")[-2] == 'heatRampSham2':
                if quad == 'croppedQ1':
                    arenafile = 'oscillator_q1.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'oscillator_q2.arena'
                scaling = 4.7
                tval = 100  # 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'exc8' or folderWithDataFiles.split("/")[-2] == 'exc66':
                if quad == 'croppedQ2':
                    arenafile = 'new_gr28_q2.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'new_gr28_q4.arena'
                scaling = 3.9
                tval = 40
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'A1_Kir' or folderWithDataFiles.split("/")[-2] == 'P9_Kir':
                if quad == 'croppedQ2':
                    arenafile = 'q2_DNs.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_DNs.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_DNs.arena'
                scaling = 3.9
                tval = 70
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'HC_Kir_cold' or folderWithDataFiles.split("/")[-2] == 'Kir_+_cold':
                if quad == 'croppedQ2':
                    arenafile = 'q2_hc_cold.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_hc_cold.arena'
                scaling = 3.9
                tval = 40
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'df_TRPA1' or folderWithDataFiles.split("/")[-2] == 'sh_kir' or folderWithDataFiles.split("/")[-2] == 'sh+' or folderWithDataFiles.split("/")[-2] == 'kir+_new':
                if quad == 'croppedQ2':
                    arenafile = 'q2_sept2019.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_sept2019.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 150
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile + ' ' +
                          outputDir + ' '+str(tval) + ' '+str(scaling))  # + ' '+str(bodyLength)+ ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'shTRPA1_Kir':
                if quad == 'croppedQ2':
                    arenafile = 'q2_shtrpa1.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_shtrpa1.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 150
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'shTRPA1+':
                if quad == 'croppedQ2':
                    arenafile = 'q2_shtrpa1.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_shtrpa1.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 150
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'newFL50':
                if quad == 'croppedQ2':
                    arenafile = 'q2_newfl50.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_newfl50.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 150
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))
            elif folderWithDataFiles.split("/")[-2] == 'DNC' or folderWithDataFiles.split("/")[-2] == 'RUT' or folderWithDataFiles.split("/")[-2][0:3] == 'day' or folderWithDataFiles.split("/")[-2] == 'FL50_check':
                if quad == 'croppedQ2':
                    arenafile = 'q2_newfl50.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_newfl50.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 80
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))
            elif folderWithDataFiles.split("/")[-2] == 'Constant' or folderWithDataFiles.split("/")[-2] == 'WT_constant':
                if quad == 'croppedQ2':
                    arenafile = 'q2_constant.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_constant.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 100
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))
            elif folderWithDataFiles.split("/")[-2] == 'both_silenced':
                if quad == 'croppedQ1':
                    arenafile = 'q1_Silence.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_Silence.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_Silence.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_Silence.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 200
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'output_40last2':
                if quad == 'croppedQ1':
                    arenafile = 'q1_late2016.arena'
                elif quad == 'croppedQ2':
                    arenafile = 'q2_late2016.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_late2016.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_late2016.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 200
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))
            elif folderWithDataFiles.split("/")[-2] == 'WT_15_cropped' or folderWithDataFiles.split("/")[-2] == 'WT_20_cropped' or folderWithDataFiles.split("/")[-2] == 'Ablated_15' or folderWithDataFiles.split("/")[-2] == 'Ablated_20':
                if quad == 'croppedQ2':
                    arenafile = 'q2_cold.arena'
                elif quad == 'croppedQ3':
                    arenafile = 'q3_cold.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_cold.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 100
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))

            elif folderWithDataFiles.split("/")[-2] == 'nTRPA1':
                if quad == 'croppedQ2':
                    arenafile = 'q2_newfl50.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_newfl50.arena'
                bodyLength = 14
                antennaeDist = 0.7
                scaling = 4.7
                tval = 100
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile + ' 0 ' + arenafile +
                          ' ' + outputDir + ' '+str(tval) + ' '+str(scaling) + ' '+str(bodyLength) + ' '+str(antennaeDist))
            elif folderWithDataFiles.split("/")[-2] == 'DNCf' or folderWithDataFiles.split("/")[-2] == 'RUTf':
                if quad == 'croppedQ2':
                    arenafile = 'q2_learning_2020.arena'
                elif quad == 'croppedQ4':
                    arenafile = 'q4_learning_2020.arena'
                scaling = 4.7
                tval = 80
                arenafile = 'arena/'+arenafile
                dataFile = filename.split('.')[0] + ".mp4"
                print(arenafile)
                os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile +
                          ' 0 ' + arenafile + ' ' + outputDir + ' '+str(tval) + ' '+str(scaling))
            """






                          
            # elif folderWithDataFiles.split("/")[-2] == 'TRPA1':

            # 	# if filename.split("_")[2].split('-')[-1] =='2017':
            # 	# 	if quad == 'croppedQ1':
            # 	# 		arenafile = 'q1_Nov2017.arena'
            # 	# 	elif quad == 'croppedQ2':
            # 	# 		arenafile = 'q2_Nov2017.arena'
            # 	# 	elif quad == 'croppedQ3':
            # 	# 		arenafile = 'q3_Nov2017.arena'
            # 	# 	elif quad == 'croppedQ4':
            # 	# 		arenafile = 'q4_Nov2017.arena'
            # 	# 	tval = 200
            # 	if filename.split("_")[2].split('-')[-1] =='2018':
            # 		if quad == 'croppedQ1':
            # 			arenafile = 'q1_trpa.arena'
            # 		elif quad == 'croppedQ2':
            # 			arenafile = 'q2_trpa.arena'
            # 		elif quad == 'croppedQ3':
            # 			arenafile = 'q3_trpa.arena'
            # 		elif quad == 'croppedQ4':
            # 			arenafile = 'q4_trpa.arena'
            # 		tval = 125

            # 		print filename.split("_")[3].split('-')[-1]
            # 		print folderWithDataFiles
            # 		print arenafile
            # 		arenafile = 'arena/'+arenafile
            # 		dataFile = filename.split('.')[0] + ".mp4"
            # 		#print dataFile
            # elif folderWithDataFiles.split("/")[-2] == 'FL50_2018':
            # 	if quad == 'croppedQ1':
            # 		arenafile = 'q1_FL502018.arena'
            # 	elif quad == 'croppedQ2':
            # 		arenafile = 'q2_FL502018.arena'
            # 	elif quad == 'croppedQ3':
            # 		arenafile = 'q3_FL502018.arena'
            # 	elif quad == 'croppedQ4':
            # 		arenafile = 'q4_FL502018.arena'
            # 		bodyLength = 11.6
            # 		antennaeDist = 0.58
            # 		scaling=3.9
            # 		tval=80
            # 		arenafile = 'arena/'+arenafile
            # 		dataFile = filename.split('.')[0] + ".mp4"
            # 		os.system("python trackImproved_1.py " + folderWithDataFiles+'/'+dataFile+ ' 0 ' + arenafile +' '+ outputDir + ' '+str(tval)+ ' '+str(scaling)+ ' '+str(bodyLength)+ ' '+str(antennaeDist))
