import ROOT
import numpy as np
import pandas as pd
import printColor as pc
color = pc.bcolors()

ROOT.ROOT.EnableImplicitMT()

def importROOT(filename):
	f = ROOT.TFile.Open(filename, "read")
	Hit = f.Get("Hit")
	dataTrack, columnsTrack = Hit.AsMatrix(return_labels=True)
	track = pd.DataFrame(data=dataTrack, columns=columnsTrack)
	#print(track)
	return(track)

def dataWindowing(track, timeWinTotal, timeWinOne):
    print(color.GREEN + "dataWindowing is Starting.."  + color.ENDC)
    #timeWinTotal = 60e-6 # s
    #timeWinOne = 5e-8 # s
    timeWinStep = timeWinOne * 1e+6 #us
    timeWinNum = int(timeWinTotal / timeWinOne)
    print(color.GREEN + "data name is ", track , " is descreting.."  + color.ENDC)
    print(color.GREEN + "Windows Number : ", timeWinNum, "." + color.ENDC)
    timeNum = range(timeWinNum)
    dfMuonPos = pd.DataFrame([])
    for j in timeNum:
        dfMuonPosTemp = track[(track['hitTime'] >= j*timeWinStep) & (track['hitTime'] < (j+1)*timeWinStep)] 
        dfMuonPosTemp['timeBin'] = j
        if (j % 100 ==0):
            print(j, "th window:", j*timeWinStep, " ~ ", j * timeWinStep + timeWinStep ,  "[us] is creating..")
        dfMuonPosTemp = dfMuonPosTemp[['eventID', 'hitTime' , 'hitPosX', 'hitPosY' ,'hitPosZ','hitPMag','hitR','eDep','hitAngle','VolID','timeBin']]
        dfMuonPos = pd.concat([dfMuonPos, dfMuonPosTemp])
    return dfMuonPos
    print("dataWindowing is done..")

def dataMomCut(track, momMin, momMax):
    print("Momentum cutting.", momMin , "~" , momMax, "Range Data is selected.")
    dfMomCut = track[(track['hitPMag'] >= momMin) & (track['hitPMag'] <= momMax)] 
    return dfMomCut


