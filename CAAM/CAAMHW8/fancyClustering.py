#Spectral Clustering:
#By a grpah, we mean a set of objects in which some pairs of the objects are "related" in some sense
#E.g. edges? pairwise relations ? Represent objects Wait is that how ??!?!?!??!?1/


#Google Page rank ? ?!???!??!?@/!/ --> Directed grpah, not undirected, links between one page, ut not back to another. 

#Ok, Adjacency matrix - (undirected graph) - 
"""
Discussion, plan, outline/the like:

Ok: First, let's just focus on trying to be able to load an image into python - baby steps, here, we've never worked with an image library before:

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipympl
#import imageio.v3 as iio
import skimage
import skimage.color
import skimage.transform
import skimage.util

def main():
    #Ok this just reads 
    chair = mpimg.imread('/Users/richard/repo/python-playground/CAAM/CAAMHW8/images/thresholding.jpeg')
    figure, ax = plt.subplots() # This generates like plots + axes ?
    plt.imshow(chair)
    plt.show()
    print(chair)
    
    # set the random seed so we all get the same matrix
    pseudorandomizer = np.random.RandomState(2021)
    # create a 4 × 4 checkerboard of random colours
    checkerboard = pseudorandomizer.randint(0, 255, size=(4, 4, 3))
    red_channel = checkerboard * [0, 1, 0]
    # restore the default map as you show the image
    plt.imshow(red_channel)
    plt.show()
    # display the arrays
    print(checkerboard)

    #The key to this is to impelmetn the connected component analysis algorithm.
    
    
    return 0
    #plt.show()

    #plt.imshow(chair) #This should show an image.

    
    #iio.imwrite(uri="CAAMHW8/images/thresholding.jpg", image=chair)
    

main()