import os
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib  import pyplot
from mtcnn.mtcnn import MTCNN


class FaceData:

    def __init__ (self, root_dir, required_size):
        self.root_dir = root_dir
        self.required_size = required_size


    def get_faceImageArray(self, path, required_size):
        
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
            image = image.resize(self.required_size)         #get the size you want
            face_array = asarray(image)                 #convert into array format
            return face_array

        except Exception as e:
            print(e)
            pass                                        

            '''Some image may not resized into the give dimension 
            hence using try and except. Here odd case is simply being 
            'pass' '''


    def get_allFaces(self):
        for path in os.listdir(self.root_dir):
            full_path = root_dir + '/' + path               #get full path of each image
            face = self.get_faceImageArray(full_path, self.required_size)
            print(face.shape)

            pyplot.imshow(face)

            ''' To get only one out of all, break is used,
            otherwise, break should be removed'''

            break
        pyplot.show()
    

root_dir = '/home/kishor/vsc/ilab/Face_recog/test_image'
f = FaceData(root_dir, (160,160))
f.get_allFaces()
