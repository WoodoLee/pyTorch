import ROOT
import numpy as np
import pandas as pd
import printColor as pc
import scipy.stats as stats
import itertools
color = pc.bcolors()
import pickle
import winPlot as plotting
print (color.HEADER + "=========================================================="+ color.ENDC)
print(color.GREEN +"pickle is opening.."+ color.ENDC)
print (color.HEADER + "=========================================================="+ color.ENDC)
pkData = open("../../../../data/rnn_data/pkData.pkl", "rb")
pkDataWin = open("../../../../data/rnn_data/pkDataWin.pkl", "rb")
pkDataCut = open("../../../../data/rnn_data/pkDataCut.pkl", "rb")
pkDataCutWin = open("../../../../data/rnn_data/pkDataCutWin.pkl", "rb")

dfData = pickle.load(pkData)
dfDataWin = pickle.load(pkDataWin)
dfDataCut = pickle.load(pkDataCut)
dfDataCutWin = pickle.load(pkDataCutWin)

pkData.close()
pkDataWin.close()
pkDataCut.close()
pkDataCutWin.close()
print (color.HEADER + "=========================================================="+ color.ENDC)
print(color.RED + "pickle Data is closed.."  + color.ENDC)
print (color.HEADER + "=========================================================="+ color.ENDC)

def timeBinSel(data, timeBinMin, timeBinMax):
    dfDataSum = pd.DataFrame([])
    dfDataSum = data[(data["timeBin"] >= timeBinMin) & (data["timeBin"] < timeBinMax)]
    return dfDataSum

timeBinMin = 0
timeBinMax = 200


dfDataSel = timeBinSel(dfDataWin, timeBinMin, timeBinMax)
dfDataSelCut = timeBinSel(dfDataCutWin, timeBinMin, timeBinMax)

dfDataSelHitCut  = 6
dfDataSelCutHitCut  = 3

plotting.rawDataPlot(timeBinMin,timeBinMax, dfDataSel, dfDataSelCut, dfDataSelHitCut, dfDataSelCutHitCut)


