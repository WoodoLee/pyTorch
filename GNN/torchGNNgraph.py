import time
import random
import ROOT
import numpy as np
import pandas as pd
import moduleGNN
import printColor as pc
import math
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
#build a graph for GNN
file1 = "~/data/g2wd10k_1.root"
rCut = 0

momCutmin = 0
momCutmax = 300

def selectData(file, momCutmin, momCutmax, rCut, labelID):
    LabelData = modules.momLabel(file, momCutmin, momCutmax, rCut)
    LabelDataSel = LabelData[(LabelData['label'] == labelID)]
    print(color.GREEN + "..", len(LabelData["label"]), "- s track is in this label", color.ENDC )
    return LabelDataSel

dataTrkTrS = moduleGNN.momRCut(file1,  momCutmin, momCutmax, rCut)

def selectRanSample(data, number):
    eveid = data.drop_duplicates('eventID')['eventID']
    dataTrkSingleMin = eveid.min()
    dataTrkSingleMax = eveid.max()
    dataTrkSingleRange = range(int(dataTrkSingleMin) , int(dataTrkSingleMax)+1)
    dataTrkSampleLabel = random.sample(dataTrkSingleRange , number)
    return dataTrkSampleLabel

sampleN = 2
sampleTr = selectRanSample(dataTrkTrS, sampleN)
#sampleTe = selectRanSample(dataTrkTeS, sampleN)
dfHitCut= moduleGNN.momRCut(file1, momCutmin, momCutmax, rCut)

#Selecting Eve ID
eveId = dfHitCut.drop_duplicates('eventID')['eventID']
#eveidTe = dfHitCutTe.drop_duplicates('eventID')['eventID']
print("event ID = ", eveId )

print("Total of eventID  =", len(eveId))
eID1 = eveId.values[110]
eID2 = eveId.values[220]
eID3 = eveId.values[330]
eID4 = eveId.values[44]

sampleeID = (int(eID1) , int(eID2), int(eID3))

momRange = range(momCutmin , momCutmax)

print(momRange)
LabelData = moduleGNN.momLabel(file1, momCutmin, momCutmax, rCut)
    
dataTrkTr1 = dfHitCut[(dfHitCut['eventID'] == eID1)]
dataTrkTr2 = dfHitCut[(dfHitCut['eventID'] == eID2)]
dataTrkTr3 = dfHitCut[(dfHitCut['eventID'] == eID3)]

def dataPre(Track1, Track2, Track3):
    TrackSum = pd.concat([Track1, Track2,Track3])
    #class_le = LabelEncoder()
    z = TrackSum['hitPosZ'].values
    t = TrackSum['hitTime'].values
    
    z = z.tolist()
    t = t.tolist()
    #TrackSum = TrackSum[TrackSum.columns.difference(['eventID'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitMag'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitAngle'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitR'])]
    size = len(TrackSum.index)
    return TrackSum, t , z, size

def dataTZ(Track):
    z = Track['hitPosZ'].values
    t = Track['hitTime'].values
    z = z.tolist()
    t = t.tolist()
    size = len(Track.columns)
    return t, z, size


def dataVZ(Track):
    #v = Track['VolID'].values
    #z = Track['hitPosZ'].values
    v = Track['VolID']
    z = Track['hitPosZ']
    
    #v = v.tolist()
    #z = z.tolist()
    #vz = Track[['VolID', 'hitPosZ']].values
    vz = Track[['VolID', 'hitPosZ']]
    #vz = vz.tolist()
    size = len(Track.columns)
    return vz, v, z, size

def dataCompSum(c1, c2):
	cSum = pd.concat([c1,c2])
	print(cSum)
	return cSum

trackSum, t_trainSum , z_trainSum , size_trackGraph = dataPre(dataTrkTr1, dataTrkTr2, dataTrkTr3)   

trackT1 , trackZ1, trackTZSize1 = dataTZ(dataTrkTr1)
trackT2 , trackZ2, trackTZSize2 = dataTZ(dataTrkTr2)
trackT3 , trackZ3, trackTZSize3 = dataTZ(dataTrkTr3)

trackVZ1, trackV1 , trackZ1, trackVZSize1 = dataVZ(dataTrkTr1)
trackVZ2, trackV2 , trackZ2, trackVZSize2 = dataVZ(dataTrkTr2)
trackVZ3, trackV3 , trackZ3, trackVZSize3 = dataVZ(dataTrkTr3)

trackVZs = dataVZ(trackSum)

print(trackVZ1)
print(len(trackVZ1))
trackVSum = dataCompSum(trackV1, trackV2)
print(trackVSum)

def buildGraph(TrackList):
    pointsN = len(TrackList)
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    
    g.add_nodes(pointsN)
    
    # all 78 edges as a list of tuples
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*TrackList))
    print(src)
    print(dst)
    g.add_edges(src, dst)
    print("==================== TEST LINE ====================")

    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)
    print(g)
    return g

test = buildGraph(trackVZ1)
#

#if __name__ == '__main__':
#	print(trackVZ1)
#    G = buildGraph(trackVZ1)
#    print('%d nodes.' % G.number_of_nodes())
#    print('%d edges.' % G.number_of_edges())
#
#    fig, ax = plt.subplots()
#    fig.set_tight_layout(False)
#    nx_G = G.to_networkx().to_undirected()
#    pos = nx.kamada_kawai_layout(nx_G)
#    nx.draw(nx_G, pos, with_labels=True, node_color=[[0.5, 0.5, 0.5]])
#    plt.show()
#
#    # assign features to nodes or edges
#    G.ndata['feat'] = torch.eye(34)
#    print(G.nodes[2].data['feat'])
#    print(G.nodes[1, 2].data['feat'])
#