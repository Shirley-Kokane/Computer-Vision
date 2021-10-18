import os, multiprocessing
from os.path import join, isfile
from multiprocessing import Pool
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from opts import get_opts
import random
from tqdm import tqdm
#%matplotlib inline

import skimage.color


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    if img.shape[2] == 1: 
          img = np.matlib.repmat(img,1,1,3)
    
    lab_img = skimage.color.rgb2lab(img)
    
    filter_img = np.zeros((int(lab_img.shape[0]), int(lab_img.shape[1]), 60))
    for i in range(0, lab_img.shape[2]):
        first= scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma = 1)
        filter_img[:,:,i] = first
        second= scipy.ndimage.gaussian_laplace(lab_img[:,:,i],sigma=1)
        filter_img[:,:,3+i]    =second
        third = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=1, order = (0,1))
        filter_img[:,:,6+i] = third
        fourth = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=1, order= (1,0))
        filter_img[:,:,9+i] = fourth
        first= scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma = 2)
        filter_img[:,:,12+i] = first
        second= scipy.ndimage.gaussian_laplace(lab_img[:,:,i],sigma=2)
        filter_img[:,:,15+i]    =second
        third = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=2, order = (0,1))
        filter_img[:,:,18+i] = third
        fourth = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=2, order= (1,0))
        filter_img[:,:,21+i] = fourth
        first= scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma = 4)
        filter_img[:,:,24+i] = first
        second= scipy.ndimage.gaussian_laplace(lab_img[:,:,i],sigma=4)
        filter_img[:,:,27+i]    =second
        third = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=4, order = (0,1))
        filter_img[:,:,30+i] = third
        fourth = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=4, order= (1,0))
        filter_img[:,:,33+i] = fourth
        first= scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma = 8)
        filter_img[:,:,36+i] = first
        second= scipy.ndimage.gaussian_laplace(lab_img[:,:,i],sigma=8)
        filter_img[:,:,39+i]    =second
        third = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=8, order = (0,1))
        filter_img[:,:,42+i] = third
        fourth = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=8, order= (1,0))
        filter_img[:,:,45+i] = fourth
        first= scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma = np.sqrt(2))
        filter_img[:,:,48+i] = first
        second= scipy.ndimage.gaussian_laplace(lab_img[:,:,i],sigma= np.sqrt(2))
        filter_img[:,:,51+i]    =second
        third = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=np.sqrt(2), order = (0,1))
        filter_img[:,:,54+i] = third
        fourth = scipy.ndimage.gaussian_filter(lab_img[:,:,i], sigma=np.sqrt(2), order= (1,0))
        filter_img[:,:,57+i] = fourth
    #Image.fromarray(img*255, 'RGB').show()
    #filtered_image.show()
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    return filter_img

def extract_filter_responses_old(opts, image):
	'''
	Extracts the filter responses for the given image.
	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	[m,n,channel] = np.shape(image)

	# make sure that entries in image are float and with range 0 1
	if (type(image[0,0,0]) == int ):
		image = image.astype('float') / 255
	elif (np.amax(image) > 1.0):
		image = image.astype('float') / 255

	if channel == 1: # grey
		image = np.matlib.repmat(image,1,1,3)
	if channel == 4: # special case
		image = image[:,:,0:3]
	channel = 3
	image = skimage.color.rgb2lab(image)
	scale = [1,2,4,8,8 * np.sqrt(2)]
	F = len(scale) * 4
	response = np.zeros((m, n, 3*F))
	#for i in range(channel):
	#	for j in range (len(scale)):
	#		response[:,:,i*len(scale)*4+j*4] = scipy.ndimage.gaussian_filter(image[:,:,i],sigma = scale[j],output=np.float64) # guassian
	#		response[:,:,i*len(scale)*4+j*4+1] = scipy.ndimage.gaussian_laplace(image[:,:,i],sigma = scale[j],output=np.float64) # guassian laplace
	#		response[:,:,i * len(scale)*4 + j*4+2] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order = [0,1],output = np.float64) # derivative in x direction
	#		response[:, :, i * len(scale)*4 + j*4+3] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order=[1, 0],output = np.float64)  # derivative in y direction


	# ----- TODO -----
	for i in range(channel):
		for j in range (len(scale)):
			response[:,:,channel*4*j+i] = scipy.ndimage.gaussian_filter(image[:,:,i],sigma = scale[j],output=np.float64) # guassian
			response[:,:,channel*4*j+3+i] = scipy.ndimage.gaussian_laplace(image[:,:,i],sigma = scale[j],output=np.float64) # guassian laplace
			response[:,:,channel*4*j+6+i] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order = [0,1],output = np.float64) # derivative in x direction
			response[:,:,channel*4*j+9+i] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order=[1, 0],output = np.float64)  # derivative in y direction
	return response



def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    opts = get_opts()
    index, alpha, train_files = args
    img = Image.open('../data/'+(train_files))
    img = np.array(img).astype(np.float32) / 255

    # randomly collect pixels
    filter_response = extract_filter_responses(opts, img)
    random_y = np.random.choice(filter_response.shape[0], int(alpha))
    random_x = np.random.choice(filter_response.shape[1], int(alpha))

    sub_img = filter_response[random_y, random_x, :]
    np.save(os.path.join('../temp/', str(index) + '.npy'), sub_img)
	
	

    # ----- TODO -----
    return filter_response

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    
    '''
    opts = get_opts()
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    #image_location = join(opts.data_dir, '..data/train files.txt')
    
    total_responses = np.zeros(shape = (alpha,60))
    for img_path in tqdm(train_files):
        img_path = join(opts.data_dir, img_path)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        filter_responses = extract_filter_responses(opts, img)
        #print((filter_responses))
        reshape_response = filter_responses.reshape(filter_responses.shape[0]*filter_responses.shape[1], filter_responses.shape[2])
        #print(reshape_response.shape)
        indices = np.random.choice(range(0, filter_responses.shape[0]*filter_responses.shape[1]), alpha)
        #print(len(indices))
        #print(indices.shape)
        alpha_responses = reshape_response[indices, :]
        #print(alpha_responses.shape)
        total_responses = np.concatenate((total_responses, alpha_responses), axis=0)
        #print(total_responses.shape)
        #print('end')
    kmeans = KMeans(n_clusters = K).fit(total_responses[alpha:])
    
    dictionary = kmeans.cluster_centers_

    # ----- TODO -----
    

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def compute_dictionary_old(opts, n_worker=8):
    '''
    Creates the dictionary of visual words by clustering using k-means.
    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(
        join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    alpha = opts.alpha
    img_size = len(train_files)
    img_list = np.arange(img_size)
    alpha_list = np.ones(img_size) * alpha

    worker = Pool(n_worker)
    args = list(zip(img_list, alpha_list, train_files))
    worker.map(compute_dictionary_one_image,args)

    filter_responses = []
    for i in tqdm(range(img_size)):
        temp_files = np.load('../temp/' + str(i)+'.npy')
        filter_responses.append(temp_files)

    filter_responses = np.concatenate(filter_responses, axis=0)
    kmeans = KMeans(n_clusters=K).fit(filter_responses)
    dict = kmeans.cluster_centers_

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dict)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    if len(img.shape) == 2:
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        new_img[:,:,0] = img
        new_img[:,:,1] = img
        new_img[:,:,2] = img
        img = new_img
    filter_img = extract_filter_responses(opts, img)
    new_img = np.zeros((filter_img.shape[0],filter_img.shape[1]))
    #print(new_img.shape)
    #for H in range(0, filter_img.shape[0]):
    #    for W in range(0,filter_img.shape[1]):
    reshape_img = filter_img.reshape(filter_img.shape[0]*filter_img.shape[1], 60)  
    
    eucli = scipy.spatial.distance.cdist(reshape_img, dictionary)
    eucli_line = np.argmin(eucli,axis=1)
    new_img = eucli_line.reshape(filter_img.shape[0],filter_img.shape[1])
                
    
    
    # ----- TODO -----
    return new_img

