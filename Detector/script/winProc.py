import numpy as np
import pandas as pd
import readMod as read
import pickle

filename = "~/data/g2wd10k_1.root"
fullWindow = 60e-6 #s
oneWindow = 5e-9 #

dfData   = read.importROOT(filename)
dfDataWin   = read.dataWindowing(dfData, fullWindow, oneWindow )

dfDataCut = read.dataMomCut(dfData , 200 , 275 )
dfDataCutWin   = read.dataWindowing(dfDataCut, fullWindow , oneWindow)

pkData = open("../../../../data/rnn_data/pkData.pkl", "wb")
pkDataWin = open("../../../../data/rnn_data/pkDataWin.pkl", "wb")
pkDataCut = open("../../../../data/rnn_data/pkDataCut.pkl", "wb")
pkDataCutWin = open("../../../../data/rnn_data/pkDataCutWin.pkl", "wb")

pickle.dump(dfData, pkData)
pickle.dump(dfDataWin, pkDataWin)
pickle.dump(dfDataCut, pkDataCut)
pickle.dump(dfDataCutWin, pkDataCutWin)

pkData.close()
pkDataWin.close()
pkDataCut.close()
pkDataCutWin.close()
