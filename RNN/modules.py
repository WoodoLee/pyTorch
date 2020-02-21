import gdal
import matplotlib._color_data as mcd
import ROOT
import numpy as np
import pandas as pd
from mayavi import mlab
import time
ROOT.ROOT.EnableImplicitMT()
print("reading..")

def hitPosition( name, eID, Rcut):
    f = ROOT.TFile.Open(name, "read")
    hit = f.Get("Hit")
    #dataMc, columnsMc = mc.AsMatrix(return_labels=True)
    dataHit, columnsHit = hit.AsMatrix(return_labels=True)
    #dfMc = pd.DataFrame(data=dataMc, columns=columnsMc)
    dfHit = pd.DataFrame(data=dataHit, columns=columnsHit)
    dfHit = dfHit.loc[(dfHit['hitTime'] > 0)]
    #print('R = ', R, 'type R = ', type(R))
    dfHitRCut = dfHit.loc[(dfHit['hitR'] > Rcut) & (dfHit['hitR'] < 330) ]

    x = dfHitRCut.loc[ dfHitRCut["eventID"] == eID, ["hitPosX"]]
    y = dfHitRCut.loc[ dfHitRCut["eventID"] == eID, ["hitPosY"]]
    z = dfHitRCut.loc[ dfHitRCut["eventID"] == eID, ["hitPosZ"]]
    t = dfHitRCut.loc[ dfHitRCut["eventID"] == eID, ["hitTime"]]
    r = dfHitRCut.loc[ dfHitRCut["eventID"] == eID, ["hitR"]]
    print('t type = ', type(t), t)
    #return x,y,z,r,t
    return dfHitRCut

def momCut( name, momCutmin, momCutmax):
    f = ROOT.TFile.Open(name, "read")
    hit = f.Get("Hit")
    dataHit, columnsHit = hit.AsMatrix(return_labels=True)
    dfHit = pd.DataFrame(data=dataHit, columns=columnsHit)
    dfHitCut = dfHit.loc[(dfHit['hitPMag'] > momCutmin) &  (dfHit['hitPMag'] < momCutmax)]
    #print(dfHitCut)
    eID = dfHitCut["eventID"]
    x   = dfHitCut["hitPosX"]
    y   = dfHitCut["hitPosY"]
    z   = dfHitCut["hitPosZ"]
    t   = dfHitCut["hitTime"]   
    #print(dfHitRCut)
    return dfHitRCut, x , y , z, t, eID
#print(eID)

def momRCut( name, momCutmin, momCutmax, Rcut):
    f = ROOT.TFile.Open(name, "read")
    hit = f.Get("Hit")
    #print('== test line == test line == test line == test line ==')
    dataHit, columnsHit = hit.AsMatrix(return_labels=True)
    dfHit = pd.DataFrame(data=dataHit, columns=columnsHit)
    #dfHitCut = dfHit.loc[(dfHit['hitPMag'] > momCutmin) &  (dfHit['hitPMag'] < momCutmax)]
    dfHitCut = dfHit.loc[(dfHit['hitPMag'] > momCutmin) &  (dfHit['hitPMag'] < momCutmax) &
            (dfHit['hitTime'] > 0)]
    print('dfHitCut = ', dfHitCut)
    dfHitRCut = dfHitCut.loc[(dfHitCut['hitR'] > Rcut)]
    print('dfHitRCut = ', dfHitRCut)
    #print(dfHitCut)
    eID = dfHitRCut["eventID"]
    x   = dfHitRCut["hitPosX"]
    y   = dfHitRCut["hitPosY"]
    z   = dfHitRCut["hitPosZ"]
    t   = dfHitRCut["hitTime"]
    r   = dfHitRCut["hitR"]
    return dfHitRCut, eID

def mcTime(name, eID):
    f = ROOT.TFile.Open( name, "read")
    mc = f.Get("MC")
    dataMc, columnsMc = mc.AsMatrix(return_labels=True)
    dfMc = pd.DataFrame(data=dataMc, columns=columnsMc)
    #print(dfMc)
    mcT = dfMc.loc[ dfMc["eventID"] == eID, ["mcTime"]]
    #print(mcT)
    return mcT

def pileup2(name, eID1, eID2):
    x1,y1,z1,r1,t1= hitPosition(name, eID1, Rcut)
    x2,y2,z2,r2,t2 = hitPosition(name, eID2, Rcut)
    pileX = pd.concat([x1,x2])
    pileY = pd.concat([y1,y2])
    pileZ = pd.concat([z1,z2])
    return pileX, pileY, pileZ

def circle( r , numbers):
    circle = pd.DataFrame(2 * np.pi * np.random.rand(numbers), index=range(0,numbers), columns =['theta'])
    circle['x'] = r * np.cos(circle['theta'])
    circle['y'] = r * np.sin(circle['theta'])
    #circle['z'] = 20 * circle['theta']
    circle['z'] = 0
    return circle
