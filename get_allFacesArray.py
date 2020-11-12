import os
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib  import pyplot
from mtcnn.mtcnn import MTCNN


def get_faceArray(path, required_size = (160, 160)):
        
        try:
            image = Image.open(path)                    #load image from given file
            image = image.convert('RGB')                #convert to RGB (optional)
            pixels = asarray(image)                     #covert to array datatype
            detector = MTCNN()                          #create MTCNN detector
            results = detector.detect_faces(pixels)     #detect face
            x1, y1, width, height = results[0]['box']   #get boundary of face

            ''' the MTCNN algorithm may sometime give 
            negative values of x1 and y1 hence make x1 
            and y1 to have absolute value..and it works 
            this way!'''
        
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1+width, y1+height                #get box arround face
            face = pixels[y1:y2, x1:x2]                 #extract face
            image = Image.fromarray(face)               #creates image memories from an object exploiting array functionality
            image = image.resize(required_size)         #get the size you want
            face_array = asarray(image)                 #convert into array format
            return face_array

        except Exception as e:
            print(e)
            pass                                        

'''Some image may not resized into the give dimension 
hence using try and except. Here odd case is simply being 
'pass' '''


def get_allFaces(root_dir):
    for path in os.listdir(root_dir):
        full_path = root_dir + '/' + path               #get full path of each image
        face = get_faceArray(full_path)
        print(face.shape)
        pyplot.imshow(face)

        ''' To get only one out of all, break is used,
        otherwise, break should be removed'''

        break
    pyplot.show()
    
root_dir = '/home/kishor/vsc/ilab/Face_recog/test_image'
get_allFaces(root_dir)
