import ROOT
import numpy as np
import pandas as pd
import time
import printColor as pc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from celluloid import Camera
from moviepy.editor import *
from matplotlib import gridspec
import scipy.stats as stats
import itertools
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
color = pc.bcolors()

ROOT.ROOT.EnableImplicitMT()

def importROOT(filename):
	f = ROOT.TFile.Open(filename, "read")
	Hit = f.Get("Hit")
	dataTrack, columnsTrack = Hit.AsMatrix(return_labels=True)
	track = pd.DataFrame(data=dataTrack, columns=columnsTrack)
	#print(track)
	return(track)

def dataWindowing(track):
    print(color.GREEN + "dataWindowing is Starting.." + color.ENDC)
    timeWinTotal = 60e-6 # s
    timeWinOne = 5e-8 # s
    timeWinStep = timeWinOne*1e+6 #us
    timeWinNum = int(timeWinTotal / timeWinOne)
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


