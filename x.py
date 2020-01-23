import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(r"C:\Users\onlyp\Documents\SUBLIME_TEXT_SAVES/dataset/ex2data1.txt" , header = None).values

X = data[:,0:2]
Y = data[:,2]

m = np.size(Y)