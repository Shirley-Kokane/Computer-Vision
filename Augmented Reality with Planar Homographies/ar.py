import numpy as np
import cv2
import HarryPotterize
from loadVid import loadVid
import PIL
from PIL import Image
from opts import get_opts
from matchPics import matchPics
from helper import *
from planarH import *
from tqdm import tqdm
#Import necessary functions
opts = get_opts()

book_frames = loadVid('../data/book.mov')
panda_frames = loadVid('../data/ar_source.mov')
cv_cover = Image.open('../data/cv_cover.jpg')
all_frames = []
ideal_width = cv_cover.size[0]
ideal_height = cv_cover.size[1]
#ideal_aspect = cv_cover.shape[0]/float(cv_cover.size[0])
a = np.min([len(book_frames), len(panda_frames)])
print(a, len(book_frames), len(panda_frames))
for f,pd in tqdm(zip(book_frames[:430], panda_frames[:430])):
    pd = Image.fromarray(pd, 'RGB')
    print(pd.size,f.shape)
    #pd = pd[int(pd.shape[0]*0.1):,int(pd.shape[1]*0.20):int(pd.shape[1]*0.80), :]
    width, height = pd.size   # Get dimensions
    new_width = pd.size[0]*0.5
    left = (width - new_width)/2
    top = (height - height)/2
    right = (width + new_width)/2
    bottom = (height + height)/2
    # Crop the center of the image
    pd = pd.crop((left, top, right, bottom))

    width_percent = (ideal_width/float(pd.size[0]))
    height_percent = (ideal_height/float(pd.size[1]))
    width_size = int((float(pd.size[1])* float(width_percent)))
    height_size = int((float(pd.size[0])*float(height_percent)))
    cv_cover = np.array(cv_cover)
    matches, locs1,locs2 = matchPics(f, cv_cover, opts)
    x1 = locs1[matches[:,0]]
    x2 = locs2[matches[:,1]]
    h2to1,inliers = computeH_ransac(x1,x2,opts)
    #width = pd.shape[0]
    #height = pd.shape[1]
    #pd = pd.resize((ideal_width, width_size), Image.ANTIALIAS)#pd.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)#cv2.resize(pd, (cv_cover.shape[1], cv_cover.shape[0]))
    panda_cover = pd.resize((ideal_width, ideal_height), Image.ANTIALIAS)
    panda_cover = np.array(panda_cover)
    composite_panda = compositeH(h2to1, panda_cover, f)
    all_frames.append(composite_panda)
    #Image.fromarray(composite_panda, 'RGB').show()

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (composite_panda.shape[1], composite_panda.shape[0]))

for i in range(len(all_frames)):
    out.write(all_frames[i])
out.release()

#Write script for Q3.1
