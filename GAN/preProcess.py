import tensorflow as tf
#import tensorflow_datasets as tfds
#import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#import matplotlib.image as mpimg


#These are the target final size of all of the input images
IMG_HEIGHT = 32 #We need to reduce size, this is too big.
IMG_WIDTH = 32 

#These are how large to resize the image to before cropping a IMG_WIDTH square from the image (to "jitter" the image)
JITTER_WIDTH = 286
JITTER_HEIGHT = 286

#The below are flags denoting whether or not to randomly resize and then crop the image so that some data is left out
MIRROR = False
CROP = False

#The following is the location of the source images
all_image_folder = os.path.join('.', 'raw_images')

#The following is the destination directory of performing all of these operations
target_folder = os.path.join('.', 'formatted_dataset')

# Normalizing the images to [-1, 1] - this in particular is something that I don't think we need to do quite in this python file.
#Here's how our "main function" will look like


def random_crop(input_image):
  #stacked_image = tf.stack([input_image], axis=0)
  cropped_image = tf.image.random_crop(
      input_image, size=[ IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #real_image = tf.image.resize(real_image, [height, width],
                              # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image#, real_image

@tf.function()
def random_jitter(input_image):
  if CROP:
    # Resizing to 286x286 (Or whatever JITTER_WIDTH/JITTER_HEIGHT are set to)
    input_image = resize(input_image, JITTER_WIDTH, JITTER_HEIGHT)
    # Random cropping back to 256x256
    input_image = random_crop(input_image)
  if tf.random.uniform(()) > 0.5 and MIRROR:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    #real_image = tf.image.flip_left_right(real_image)

  return input_image#, real_image

#This will be our master function that actually does everything:
def create_dataset(img_folder):
    """
    Returns an array of img data? and a something.
    Input: Path to the desired folder to create a dataset from
    Output: An array containing the image data, along with a dictionary of labels for aforementioned data
    Except I feel that I don't actually need the class labels because we aren't creating a classifier
    But it would be useful, just in case if we do eventually decide to create such a classifier, which
    is why I plan on including this capability.

    Desired Parameters:
        Resolution (e.g. 256 x 256)
        Classes Information would be included with the folder title - did not include.
       
        The above two are the most important parameters - Good, because I've already included that above.
        Should we always include all steps, or should we have booleans regarding the exact "jitters" we
        want applied to the photo?

        Nah, I think as long as we leave those as flags, it should not be too hard to modify.

    """
    #img_data_array = []
    #class_name = []
   
    for class_dir in os.listdir(img_folder):
        unique = 0
        uniqueIdentifier = 0
        if(class_dir[-1] == 'S'): #'S' FOR SKIP
           continue
        if(class_dir == '.DS_Store' or class_dir ==".gitignore"): #Skip automatically created mac files
           continue
        for file in os.listdir(os.path.join(img_folder, class_dir)):
            unique += 1
            first_period = file.find('.')
            if file[first_period + 1:].find('.') != -1: #Rename bad files
                second_period = file.find('.', first_period + 1)
                print("REMOVING BAD FILE: " + file)
                os.rename(os.path.join(img_folder, class_dir, file), os.path.join(img_folder, class_dir, "rename" + str(unique) + file[:second_period]))
                file = file[:-2]
            image_path = os.path.join(img_folder, class_dir, file)
            if image_path[-1] != 'g': #If it's not a png/jpeg/jpg file, then we skip it
                continue
            #image = np.array(Image.open(image_path))
            #image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH, 3))
            #print(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if(class_dir == 'ENLARGE'): #If the folder has the flag ENLARGE, then we want to enlarge the image
               image = cv2.resize(image, (4*IMG_HEIGHT, 4*IMG_WIDTH),interpolation = cv2.INTER_CUBIC) #This also allows for upscaling in the case that the input image is too small
            else:
               image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA) #This also allows for upscaling in the case that the input image is too small
            #cv2.imshow('resized',image)
            #For some reason, everyone wants float32 instead of ... integers or whatever
            #image = image.astype('float32')
            #Normalize to 1, -1 to fit the tanh function as opposed to regular RGB values
            #image = (image - 127.5)/127.5
            #Convert the normalized array into a tensorflow tensor for compatibility reasons
            image = tf.convert_to_tensor(image)
            # Convert both images to float32 tensors (just in case?) No I don't think that we need this
            #input_image = tf.cast(input_image, tf.float32)
            #real_image = tf.cast(real_image, tf.float32)
            image = random_jitter(image)
            #name =
            write_path = os.path.join(target_folder, class_dir, ('modified' + str(uniqueIdentifier) + '.jpg'))
            #Create all of the class folders within the result
            #Apparently tensorflow recursively automatically creates the directories if it doesn't exist
            #os.mkdir(os.path.join('dataset/', class_dir))
            #Write to file:
            #cv2.imwrite(write_path, image)
            image = tf.io.encode_jpeg(image)
            tf.io.write_file(write_path, image)
            uniqueIdentifier +=1
            #img_data_array.append(image)
            #class_name.append(dir1)
            #return img_data_array#, class_name
            unique +=1      
            
create_dataset(all_image_folder)