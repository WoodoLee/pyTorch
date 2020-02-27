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
import detectRead

filename = "~/data/eDepPhiVid/g2wd10k_1.root"

dfData = detectRead.importROOT(filename)
dfPosi = detectRead.dataWindowing(dfData)

done = detectRead.threeWindows(dfPosi, 100 , 5)

print(done)
