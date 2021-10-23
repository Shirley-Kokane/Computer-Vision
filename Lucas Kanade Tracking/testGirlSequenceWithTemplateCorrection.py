import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("./data/girlseq.npy")
rect = [280, 152, 330, 318]
rect0 = [280, 152, 330, 318]
frame_rect = np.zeros((len(seq[0,0,:]), 4))
frame_rect[0,0] = 280 
frame_rect[0,1] = 152 
frame_rect[0,2] = 330
frame_rect[0,3] = 318
new_p = np.zeros(2)
for i in range(0, len(seq[0,0,:])-1):
    It, It1 = seq[:,:,i], seq[:,:,i+1]
    
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
    #print(rect[0], rect0[0])
    all_p = p+ [rect[0] - rect0[0], rect[1]-rect0[1]]
    pstar = LucasKanade(seq[:,:,0], It1, frame_rect[0,:], threshold, num_iters, p0=all_p)
    
    #print(p)
    diff = np.linalg.norm(all_p- pstar)
    if abs(diff)<5:
        p_now = (pstar - [rect[0] - rect0[0], rect[1] - rect0[1]])
        rect[0] = rect[0] + p_now[0]
        rect[1] = rect[1] + p_now[1]
        rect[2] = rect[2] + p_now[0]
        rect[3] = rect[3] + p_now[1]
        frame_rect[i+1,0] = rect[0]
        frame_rect[i+1,1] = rect[1]
        frame_rect[i+1,2] = rect[2]
        frame_rect[i+1,3] = rect[3]
        p0 = np.zeros(2)
    else:
        frame_rect[i+1,0] = frame_rect[i,0] + p[0]
        frame_rect[i+1,1] = frame_rect[i,1] + p[1]
        frame_rect[i+1,2] = frame_rect[i,2]+ p[0]
        frame_rect[i+1,3] = frame_rect[i,3]+p[1]
        p0 = p

    width = frame_rect[i+1,2] - frame_rect[i+1,0]
    height = frame_rect[i+1,3] - frame_rect[i+1, 1]
    print(i)
    if i%20 == 0:
        print(frame_rect[i,0]+p[0],frame_rect[i,1]+ p[1])
        fig , ax = plt.subplots(1)
        ax.imshow(It1)
        dect = patches.Rectangle((frame_rect[i+1,0]+p[0],frame_rect[i+1,1]+ p[1]),width,height,edgecolor = 'r', facecolor = 'none', linewidth = 3)
        ax.add_patch(dect)
        plt.show()
    #plt.imsave('boximage.jpeg', It)
np.save('girlseqrects-wcrt.npy', frame_rect)   