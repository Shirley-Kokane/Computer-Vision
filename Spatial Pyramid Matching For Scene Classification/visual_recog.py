import os, math, multiprocessing
from os.path import join
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import visual_words
from numpy import *
import image_slicer
from opts import get_opts
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    
    
    H,W = np.shape(wordmap)
    data = wordmap.reshape(1,H*W)
    bin_num = np.linspace(0, K, num = K+1,endpoint = True)
    bin_height,bin_boundary = np.histogram(data,bins = bin_num, density = True, range= [0,bin_num])
    #define width of each column
    #standardize each column by dividing with the maximum height
    #bin_height = bin_height/float(max(bin_height))
    # ----- TODO -----
    #plt.hist(bin_boundary[:-1], bin_boundary, weights = bin_height)
    #plt.show()
    #print(bin_boundary)
    #bin_height = np.reshape(bin_height, (1,K))
    #bin_height = bin_height/np.sum(bin_height)
    return bin_height

def get_feature_from_wordmap_SPM(opts, wordmap):
    
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    opts = get_opts()
    row_length = wordmap.shape[0]
    col_length = wordmap.shape[1]
    dict_size = 10
    L = opts.L
    K = opts.K
    arr_size = int(K*(4**(L) -1)/3)
    total_arr = np.zeros((arr_size))
    count = 0
    # ----- TODO -----    
    for r in range(0,L):
        image_div = 4**(r)
        div_num = 2**(r)
        M= wordmap.shape[0]//(image_div//div_num)
        #print(row_length, M)
        N = wordmap.shape[1]//(image_div//div_num)
        tiles = [wordmap[x:x+M,y:y+N] for x in range(0,wordmap.shape[0],M) for y in range(0,wordmap.shape[1],N)]
        #print('Is it reaching?')
        #print(len(tiles))
        final_tiles = tiles[:image_div]
        #print(len(final_tiles))
        for small in final_tiles:
            #print(small.shape)
            #print(small.shape, dictionary.shape)
            hist_arr = get_feature_from_wordmap(opts,small)
            
            total_arr[count: count + hist_arr.shape[0]] = hist_arr
            count = count + K
        #print(len(total_arr))
        #print(total_arr)
    return total_arr


    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_img = visual_words.get_visual_words(opts, img,dictionary)
    hist_arr = get_feature_from_wordmap_SPM(opts, filter_img)
    

    # ----- TODO -----
    
    return hist_arr

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K
    L = opts.L
    arr_size = int(K*(4**(L) -1)/3)

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    #print(train_files[0])
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    total_features= np.array([], dtype = np.int64).reshape(0,arr_size)
    for path in tqdm(train_files):
        img_path = join(opts.data_dir, path)
        feature = get_image_feature(opts, img_path, dictionary)
        total_features = np.vstack([total_features, feature])
    
    np.save('trained_system_features.npy', total_features)
    np.save('training_labels.npy', train_labels)
    #np.savez('trained_system_features', features = total_features, labels = train_labels, dictionary = dictionary, layernum =SPM_layer_num)
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=total_features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
     )
    pass

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    
    # ----- TODO -----
    #np.min()
    #wordhist_shape = np.broadcast_to(word_hist, (histograms.shape[0], histograms.shape[1]))
    intersection = np.minimum(histograms, word_hist)
    similarity = np.sum(intersection, axis=1)
    #similarity = 1 - product
    
    return similarity    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system_features.npy'))
    train_npz = np.load('trained_system.npz')
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    features = train_npz['features']
    #train_labels = trained_system['labels']
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    #test_opts.K = dictionary.shape[0]
    #test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    trained_labels = train_npz['labels']
    test_pre = []
    count = 0
    c = 0
    confusion = np.zeros((8,8))
    # ----- TODO -----
    for path in tqdm(test_files):
        
        img_path = join(opts.data_dir, path)
        hist_arr = get_image_feature(opts,img_path, dictionary)
        similarity = distance_to_set(hist_arr, trained_system)
        #idx = np.unravel_index(np.argmax(similarity, axis=None), similarity.shape)[0]
        
        test_pre.append(np.unravel_index(np.argmax(similarity, axis=None), similarity.shape)[0])
        predict_labels = trained_labels[np.argmax(similarity)]
        true_label = test_labels[c]
        confusion[true_label, predict_labels] += 1
        c = c+1
    accurate = np.trace(confusion)/np.sum(confusion)
    
    
    return confusion, accurate