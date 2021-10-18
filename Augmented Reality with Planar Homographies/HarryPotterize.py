import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import *
from matchPics import matchPics
from helper import *
from PIL import Image
#Import necessary functions
opts = get_opts()

sigma = opts.sigma
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

#cv_desk = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2GRAY)
#print(cv_desk.shape)
#edges = corner_detection(cv_desk, sigma)
#print(edges)

matches,locs1,locs2 = matchPics(cv_desk,cv_cover, opts)

x1 = locs1[matches[:,0]]
x2 = locs2[matches[:,1]]

#print(hp_cover.shape)
#print(cv_desk.shape)
finalh,t1 = computeH_ransac(x1, x2, opts)

width = len(cv_desk[1,:,1])
height = len(cv_desk[:,1,1])

hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]))

composite_img = compositeH(finalh, hp_cover,cv_desk)
#composite_img = composite_img.astype(int)

data = Image.fromarray(composite_img,'RGB')
Image.fromarray(composite_img,'RGB').show()
data.save('hp3.png')
#composite_img.save('cmp.png')
#Write script for Q2.2.4
