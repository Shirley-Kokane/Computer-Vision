import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanadeAffine import LucasKanadeAffine
import cv2
from SubtractDominantMotion import SubtractDominantMotion
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('./data/aerialseq.npy')

for i in range(0, len(seq[0,0,:])-1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    #width1 = int(It.shape[1]*0.6)
    #height1 = int(It.shape[0]*0.6)
    #width2 = int(It1.shape[1]*0.6)
    #height2 = int(It1.shape[0]*0.6)
    #It = cv2.resize(It, (width1, height1), interpolation = cv2.INTER_AREA)
    #It1 = cv2.resize(It1, (width2, height2), interpolation = cv2.INTER_AREA)
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    indices= np.where(mask == 1)
    
    if i%30 == 0:
        pic = plt.figure()
        plt.imshow(It1, cmap='gray')
        
        fig,= plt.plot(indices[1],indices[0] ,'*')
        
        pic.savefig('aerial_frame'+str(i)+'.png', bbox_inches='tight')