'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
from submission import *
import matplotlib.pyplot as plt
from helper import *
from findM2 import *
import os.path
from mpl_toolkits.mplot3d import Axes3D

def visualize():
    M1, C1, M2, C2, F = find_M2()
    image1 = plt.imread('./data/im1.png')
    image2 = plt.imread('./data/im2.png')
    data = np.load('./data/templeCoords.npz')
    x1 = data['x1']
    y1 = data['y1']
    
    pts1 = np.hstack((x1, y1))
    pts2 = np.zeros((x1.shape[0], 2))
    for i in range(x1.shape[0]):
        x2, y2 = sub.epipolarCorrespondence(image1, image2, F, x1[i, 0], y1[i, 0])
        pts2[i, 0] = x2
        pts2[i, 1] = y2
    


    P, err = triangulate(C1, pts1, C2, pts2)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(3.4, 4.2)
    ax.set_xlim3d(-0.8, 0.6)
    plt.show()
    
    np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)


if __name__ == '__main__':
    visualize()