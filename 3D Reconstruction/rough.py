import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import os.path
from helper import epipolarMatchGUI
from helper import displayEpipolarF
from util import _singularize
from util import refineF
from scipy.ndimage import gaussian_filter
from submission import eightpoint
#from helper import displayEpipolarF
from submission import essentialMatrix
from submission import ransacF

image1 = plt.imread('./data/im1.png')
image2 = plt.imread('./data/im2.png')
data = np.load('./data/some_corresp_noisy.npz')

F , inliers= ransacF(data['pts1'], data['pts2'], np.max(image1.shape))
print(F)
displayEpipolarF(image1, image2, F)

#K =np.load('./data/intrinsics.npz')
#E = essentialMatrix(F , K['K1'], K['K2'])
#print(E)
#epipolarMatchGUI(image1, image2, F)

