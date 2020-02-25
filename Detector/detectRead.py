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


color = pc.bcolors()

ROOT.ROOT.EnableImplicitMT()

def importROOT(filename):
	f = ROOT.TFile.Open(filename, "read")
	Hit = f.Get("tree")
	dataTrack, columnsTrack = Hit.AsMatrix(return_labels=True)
	track = pd.DataFrame(data=dataTrack, columns=columnsTrack)
	#print(track)
	return(track)

track = importROOT("~/data/eDepPhiVid/g2wd10k_1.root")
eID = track['eventID']
eID = eID.drop_duplicates()
eIDNum = len(eID)

print(len(track))
eIDNum = range(eIDNum)
timeWinTotal = 60e-6 # s
binNum = 5e-9 # s
timeWinNum = int(timeWinTotal / binNum)



print(timeWinNum)
timeNum = range(timeWinNum)
#for i in eveNum:
#	print(i)
dfMuonPos = pd.DataFrame([])
#track['timeBin'] = 0
fig, ax = plt.subplots(figsize=(30, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 3]) 
camera = Camera(fig)

BinRange = range(timeWinNum)
#test = range(100)
for j in timeNum:
#for j in test:
	#ax0 = plt.subplot(gs[0])
	#print(color.RED + "Time Window = ",j*5e-3, "-", j*5e-3+5e-3 ,color.ENDC)
	dfMuonPosTemp = track[(track['dT'] >= j*5e-3) & (track['dT'] < j*5e-3+5e-3)] 
	dfMuonPosTemp['timeBin'] = j
	if (j % 10 ==0):
		print(j)
	dfMuonPosTemp = dfMuonPosTemp[['dT' , 'posX', 'posY' ,'posZ','momMag','timeBin']]
	#ax0.bar("momMag",dfMuonPosTemp['momMag'] )
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])
	ax2 = plt.subplot(gs[2])
	#print(dfMuonPosTemp)
	dfMuonPos = pd.concat([dfMuonPos, dfMuonPosTemp])
	ax0.scatter(dfMuonPosTemp["posX"],dfMuonPosTemp["posY"], color='r' )
	#print(dfMuonPos)
	ax1.hist(dfMuonPos["momMag"], range = (0,300), bins = 300)
	ax2.hist(dfMuonPos["dT"], range = (0,60), bins = timeWinNum)
	camera.snap()
	
print(dfMuonPos)

#animation = camera.animate(interval=1, blit=True)
animation = camera.animate()
animation.save("muon.mp4")

#animation.save(
#    'test.mp4',
#    dpi=100,
#    savefig_kwargs={
#        'frameon': False,
#        'pad_inches': 'tight'
#    }
#)
clip = (VideoFileClip("muon.mp4").speedx(10))
clip.write_gif("muon.gif")



#def updateTrack(num):
#    data=dfMuonPos[dfMuonPos['timeBin']==num]
#    print(data)
#    graph._offsets3d = (data.posX, data.posY, data.posZ)
#    title.set_text('3D Test, time={}'.format(num))
#data=dfMuonPos[dfMuonPos['timeBin']==0]
#graph = ax.scatter(data.posX, data.posY, data.posZ)
#ani = matplotlib.animation.FuncAnimation(fig, updateTrack, timeWinNum, interval=1, blit=False)
#ani.save('muonDecayPosition.gif', writer='imagemagick', fps=30, dpi=100)
#plt.show()
