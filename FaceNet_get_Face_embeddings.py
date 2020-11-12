import os
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib  import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed


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
    face = list()
    for path in os.listdir(root_dir):
        full_path = root_dir + '/' + path               #get full path of each image
        Face = get_faceArray(full_path)
        print(Face.shape)
        face.extend(Face)
    return face
    

def load_dataset(root_dir):
    X, y = list(), list()
    for celebrity in categories:
        full_path = root_dir + '/' + celebrity
        X.extend(get_allFaces(full_path))
        y.extend(celebrity)
    return asarray(X), asarray(y)


categories = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
dataset_type = ['train', 'val']

root_dir = '/home/kishor/vsc/ilab/Face_recog/archive(1)/data/train'
trainX, trainy = load_dataset(root_dir)
print(trainX.shape, trainy.shape)

root_dir = '/home/kishor/vsc/ilab/Face_recog/archive(1)/data/val'
testX, testy = load_dataset(root_dir)
print(testX.shape, testy.shape)

savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)