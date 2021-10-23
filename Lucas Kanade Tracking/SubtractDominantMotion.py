import numpy as np
from scipy.interpolate.fitpack2 import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.filters import median_filter
from scipy.ndimage import affine_transform
from InverseCompositionAffine import InverseCompositionAffine
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    h1,w1 = image1.shape
    x1,y1 = np.mgrid[0:w1, 0:h1]
    

    mask = np.zeros(image1.shape, dtype=bool)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    #print(M)
    #M_HOMO = np.append(M, [[0,0,1]], axis=0)
    
    image2_warp = affine_transform(image1, M, output_shape = (h1,w1))
    Difference = abs(image1 - image2_warp)
   
    mask[Difference>=0.2] = 1
    mask[image2_warp == 0] = 0
    mask = binary_erosion(mask,structure = np.ones((1,2)), iterations = 1)
    mask = binary_erosion(mask, structure = np.ones((2,1)),iterations = 1)
    mask = binary_dilation(mask, iterations = 1)

    return mask
