# Spatial-Pyramid-Matching-for-Scene-Classification

Here we firstly imitate the bag of words approach to make visual features out of the scenes. The bag-of-words (BoW) approach, has been applied to a myriad of recognition problems in computer vision. 

Firstly, we apply a set of filter to an image. The filters are: (1) Gaussian, (2) Laplacian of Gaussian, (3) derivative of Gaussian in the x direction, and (4) derivative of Gaussian in the y direction. 

![filter_bank](https://user-images.githubusercontent.com/69525348/136774115-cf0fd727-2ae9-4108-812d-93cd52e38ebf.png)


Then we will create a dictionary of visual words from the filter responses using k-means clusters. After applying k-means, similar filter responses will be represented by the same visual word. If there are T training images, then you should collect a matrix filter responses over all the images that is alpha*T x 3F, _where F is the filter bank size. Then, to generate a visual words dictionary with K words. 

Now, we will map each pixel in the image to its closest word in the dictionary. 

![visual words](https://user-images.githubusercontent.com/69525348/136776094-9aca1321-ba28-4d23-a3de-8a248c86000e.jpg)


For the recoginiton system, we shall use the K-nearest neighbour algorithm. The key components of any nearest-neighbor system are features and similarity. The final outline of this method is as follows: 
![brief](https://user-images.githubusercontent.com/69525348/136926507-2cc691f4-d735-483a-80f2-b1e0768af80c.jpg)
