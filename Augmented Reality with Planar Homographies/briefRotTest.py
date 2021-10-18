import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
hist = []

for i in tqdm(range(36)):
	#Rotate Image
	
	img = scipy.ndimage.rotate(cv_cover, i*10)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, img, opts)

	#Update histogram
	hist.append(matches.shape[0])


hist_arr = np.histogram(hist)

fig,ax = plt.subplots(figsize= (10,5))
ax.hist(hist)

plt.show()


	#pass # comment out when code is ready


#Display histogram

