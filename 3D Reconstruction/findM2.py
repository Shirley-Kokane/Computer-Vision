'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
from helper import camera2
from submission import eightpoint, essentialMatrix, triangulate
import submission
import matplotlib.pyplot as plt
import helper
import os.path
import math

def find_M2 ():
    K_values = np.load('./data/intrinsics.npz')
    k1 = K_values['K1']
    k2 = K_values['K2']
    data = np.load('./data/some_corresp.npz')
    
    imag1 = plt.imread('./data/im1.png')
    imag2 = plt.imread('./data/im2.png')

    M = np.max(imag2.shape)

    F = eightpoint(data['pts1'], data['pts2'], M)

    E = essentialMatrix(F, k1, k2)

    print(E.shape)

    M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    M2S = camera2(E)
    
    print(M2S.shape)

    C1 = np.matmul(k1,M1)
    min_err = []
    for i in range(M2S.shape[2]):
        M2 = M2S[:,:,i]
        C2 = np.matmul(k2,M2)
        P, err = triangulate(C1, data['pts1'], C2, data['pts2'])
        min_err.append(err)
        if (np.all(P[:,2] > 0)) :
            break

    #t = min_err.index(min(min_err))
    #M2 = M2S[:,:,t]
    #C2 = np.matmul(k2, M2)
    #P, err = triangulate(C1 , data['pts1'], C2, data['pts2'])


    np.savez('q3_3.npz', M2 = M2, C2 = np.matmul(k2, M2), P = P)
    print("works")
    
    return M1,C1,M2,C2,F

if __name__ == '__main__':
    find_M2()