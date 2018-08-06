# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:00:05 2018

@author: smita
"""

#python related imports
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import json
import argparse
from collections import OrderedDict
#pytorch related imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset, DataLoader
from train import load_labels,pretrained_model


def get_commandlineArgs():
    """
    Function to parse command line arguments
    :return: processed arguments
    
    """
    parser = argparse.ArgumentParser(description=" Predict the Image")
    parser.add_argument('image', type=str, help='Image to predict')
    parser.add_argument('checkpoint', type=str,default='checkpoint.pth', help='Use saved model checkpoint to predict')
    parser.add_argument('--top_k', type=int,default=5, help='Return the top K predictions')
    parser.add_argument('--category_names', type=str, help='JSON file containing image names')
    parser.add_argument('--gpu', action='store_true', help='Compute on GPU if available')
    return parser.parse_args()



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    img_processed = transformations(img)
    
    return img_processed


def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Turn on Cuda if available
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to('cpu')
    # Process image into inputs and run through model
    img = process_image(image_path).unsqueeze(0)
    outputs = model.forward(img)
    
    
    # Get Probabilities and Classes
    print("top_k",top_k)
    probs, classes = outputs.topk(top_k)
    probs = probs.exp().data.numpy()[0]
    classes = classes.data.numpy()[0]
    class_keys = {x:y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in classes]
    return probs, classes

def get_predict_results(path):
    """
    This function invokes process_image function to get the prediction results
    in the form of top_k predicted probabilities and classes
    """
# Get and process a Test Image
    data = path.split("/")
    test_image_index = data[2]
    test_image = path
    img = process_image(test_image)
    cat_to_name = load_labels()
    # Display test image, with Label as title
    label = cat_to_name.get(str(test_image_index))
    
    print("\n\n\n\t\t\tActual Label =\t:",label)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        print("GPU available training neural network in GPU mode ...... !!!!")
    else:
        print("GPU not available .....Training in CPU mode !!!!")
    
    # Run image through model
    probs, classes = predict(test_image, model,args.top_k)
    print("\n\t\t\tProbabilies: {}".format(probs))
    print("\n\t\t\tClasses: {}".format(classes))
    y = np.arange(len(classes))
    y_labels = [cat_to_name.get(i) for i in classes[::-1]]
    x = np.array(probs)
    y_labels.reverse()
    print("\nTop Predicted Labels :\t",y_labels)
    
def load_model(checkpoint):
    model = torch.load('checkpoint.pth',map_location='cpu')
    arch = model['arch']
    class_idx = model['class_to_idx']
    print("Chosen pretrained architecture is :",arch)
    #download the pretrained model based on arch saved in checkpoint.pth file
    if arch == 'vgg19':
        mymodel = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        mymodel = models.alexnet(pretrained=True)
    elif arch == 'densenet121':
        mymodel = models.densenet121(pretrained=True)
    else:
        print('{} Pretrained architecture not found. \n\nSupported Architectures are args: \'vgg19\', \'alexnet\', or \'densenet121\''.format(arch))
        
    
    mymodel.classifier = model['classifier']
    mymodel.load_state_dict(model['state_dict'])
    mymodel.class_to_idx = model['class_to_idx']
    return mymodel    
    
    
if __name__ == '__main__':
    args = get_commandlineArgs()
    #if args.image:
    image = args.image     
        
    #if args.checkpoint:
    checkpoint = args.checkpoint
    #print("check=",checkpoint)

    #if args.top_k:
    top_k = args.top_k
            
    if args.category_names:
        category_names = args.category_names

    if args.gpu:
        gpu = args.gpu 
    
    data_dir = 'flowers/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    model = load_model(checkpoint)
    get_predict_results(image)
       