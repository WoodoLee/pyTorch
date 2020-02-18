import gdal
import matplotlib._color_data as mcd
import ROOT
import numpy as np
import pandas as pd
from mayavi import mlab
import time
import printColor as pc
color = pc.bcolors()

ROOT.ROOT.EnableImplicitMT()
def momLabel(name, momCutmin, momCutmax,Rcut):
#def momLabel(name, momCutmin, momCutmax,Rcut, labelID):
    f = ROOT.TFile.Open(name, "read")
    hit = f.Get("Hit")
    #print('== test line == test line == test line == test line ==')
    dataHit, columnsHit = hit.AsMatrix(return_labels=True)
    dfHit = pd.DataFrame(data=dataHit, columns=columnsHit)
    #dfHitCut = dfHit.loc[(dfHit['hitPMag'] > momCutmin) &  (dfHit['hitPMag'] < momCutmax)]
    momRange = range(momCutmin , momCutmax) 
    #label = list( range( len(momRange) ) )
    #label = range(0, len(momRange)
    #print(momRange)
    label = momRange
    dfHitLabel = pd.DataFrame([])
    print(color.RED + "labeling", color.ENDC , end='')
    print("label", label)
    for i in momRange:
        #print(color.RED + "starting Labeling..", label, "-th data is being labeled..", color.ENDC )
        print(color.RED + ".", color.ENDC, end='' )
        #dfHit["label"] = label
        dfHit["label"] = i
        #print(color.RED +  "momCut =  ", i , "MeV/c ~ ", i+1 ,"MeV/c" ,"...", color.ENDC )
        dfHitLabelTemp = dfHit[(dfHit['hitPMag'] > i) &  (dfHit['hitPMag'] < i+1) & (dfHit['hitTime'] > 0) &  (dfHit['hitR'] < 330) ]
        dfHitLabelTemp = dfHitLabelTemp[['eventID', 'hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle', 'eDep', 'label']]
        #dfHitLabelTemp = dfHitLabelTemp[['hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle', 'label']]
        #dfHitLabelRef = dfHitLabelTemp.iloc[0]
        #dfHitLabelRef = dfHitLabelRef[['hitTime','hitPosX', 'hitPosY', 'hitPosZ', 'hitAngle']]
        
        #refer = dfHitLabelRef.to_numpy()        
        
        #dfHitLabelTemp = dfHitLabelTemp['hitTime'] - refer[0]
        #dfHitLabelTemp = dfHitLabelTemp['hitPosX'] - refer[1]
        #dfHitLabelTemp = dfHitLabelTemp['hitPosY'] - refer[2]
        #dfHitLabelTemp = dfHitLabelTemp['hitPosZ'] - refer[3]
        #dfHitLabelTemp = dfHitLabelTemp['hitAngle'] - refer[4]

        #print(dfHitLabelTemp)
        #print(refer)
        
        #dfHitLabelTemp = dfHitLabelTemp - dfHitLabelRef
        
        #print(dfHitLabelRef)
        dfHitLabel = pd.concat([dfHitLabel , dfHitLabelTemp])
    #print(label)
    dfHitLabelCalib = pd.DataFrame([])
    #label = range(0 ,label)
    print(color.RED + " is done", color.ENDC )
    for i in label:
        dfHitLabelLen = dfHitLabel[(dfHitLabel['label'] == i)]
        if len(dfHitLabelLen) == 0:
            continue
        dfLen = len(dfHitLabelLen)
        #dfRefer = dfHitLabelLen[['hitPosX', 'hitPosY', 'hitPosZ']]
        #dfRefer = dfRefer.iloc[0].values
        dfRefer = dfHitLabelLen.iloc[0]
        #dfRefer['hitTime'] = 0.
        #dfRefer['hitR'] = 0.
        #dfRefer['hitAngle'] = 0.
        dfRefer['label'] = 0
        dfRefer['eDep'] = 0.
        dfRefer['eventID'] = 0.
        #print('dfRefer')
        #print(dfRefer)
        dfRefer = dfRefer.values
        dfHitRefer =  [dfRefer] * dfLen
        dfHitRefer = pd.DataFrame(dfHitRefer)
        dfHitLabelLen = dfHitLabelLen.values
        #dfHitLabelLen = dfHitLabelLen[['hitPosX', 'hitPosY', 'hitPosZ']].values
        dfHitLabelLen = pd.DataFrame(dfHitLabelLen)    
        #dfHitRefer.columns = ['hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle', 'label']
        dfHitRefer.columns = ['eventID','hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle','eDep', 'label']
        #dfHitLabelLen.columns = ['hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle' ,'label']
        dfHitLabelLen.columns = ['eventID','hitTime','hitPosX', 'hitPosY', 'hitPosZ','hitR', 'hitAngle','eDep' ,'label']
        dfHitLabelCalibTemp = dfHitLabelLen - dfHitRefer
        dfHitLabelCalib = pd.concat([dfHitLabelCalib, dfHitLabelCalibTemp])
        #print(color.YELLOW + "How many Tracks?", i , " -th labeled Tracks ", i ,"MeV/c ~ ", i+1 ,"MeV/c" ,"...", len(dfHitLabelLen) , color.ENDC )
        #print(dfHitRefer)
        #print(color.YELLOW + "How many Tracks?", i , " -th labeled Tracks ", i ,"MeV/c ~ ", i+1 ,"MeV/c" ,"...", len(dfHitLabelLen) , color.ENDC )
        #print(dfHitLabelLen)
        #print(color.YELLOW + "How many Tracks?", i , " -th labeled Tracks ", i ,"MeV/c ~ ", i+1 ,"MeV/c" ,"...", len(dfHitLabelLen) , color.ENDC )
        #print(dfHitLabelCalib)
            #print(color.RED + "..is done..returning values..", color.ENDC)
    #print(color.BLUE + "..", label , color.ENDC)
    #print("dfHitLabel", dfHitLabel)
    print('dfHitLabelCalib')
    print(dfHitLabelCalib)

    dfHitLabelCalib = dfHitLabelCalib[1:]
    #dfRefer = dfHitLabel.iloc[0].values
    #print('dfTest')
    #print(dfTest)
    #print('dfHitLabelCalib')
    #print(dfHitLabelCalib)
    return dfHitLabelCalib
