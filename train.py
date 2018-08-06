#python related imports
#import sys
import os
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

def get_commandlineArgs():
    parser = argparse.ArgumentParser(description=" Train a new network on a dataset and save the model as a checkpoint")

    parser.add_argument('data_dir',
                        type=str,
                        help='Path to input data directory'
                        )
    
    parser.add_argument('--save_dir',
                        type=str,
                        #default='',
                        help='Path for saving model checkpoint')
    
    parser.add_argument('--learning_rate',
                        default=0.01,
                        type=float,
                        help='Learning rate for the model')
                       
    
    parser.add_argument('--hidden_units',
                        default = '512',
                        type=str,
                        help='Number of hidden layers')
                        

    parser.add_argument('--arch',
                        default='densenet121',
                        type=str,
                        help='Specify the pretrained architecture{densenet121,vgg19,alexnet}')

    parser.add_argument('--epochs',
                        default=3,
                        type=int,
                        help='Number of epochs to train the model')
    
    parser.add_argument('--gpu',
                        default=False,
                        help='Train the model on gpu or cpu')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='Save trained model to file')

    return parser.parse_args()

def data_transformations():
    """
    This function is used to compute data transformations on
    training, testing and validation sets
    :return: traning data transformation and testing,validation transformations
    """
    training_data_transforms = transforms.Compose([transforms.RandomRotation(40),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_validation_data_transforms = transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return training_data_transforms,test_validation_data_transforms

def data_loaders():
    """
    This function is used to define the data loaders for - train, test and validation
    :return: train,validation and test dataloaders
    """
    train_image_datasets = datasets.ImageFolder(train_dir,transform=training_data_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir,transform=test_validation_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=test_validation_data_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = DataLoader(train_image_datasets,batch_size=32,shuffle=True)
    dataloaders_validation = DataLoader(validation_image_datasets,batch_size=32,shuffle=True)
    dataloaders_test = DataLoader(test_image_datasets,batch_size=32,shuffle=True)
    
    return dataloaders_train,dataloaders_validation,dataloaders_test,train_image_datasets
def load_labels():
    """
    This function loads the labels associated with images
    :return: labels
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def pretrained_model(arch='densenet121'):
    """
    This function loads the pretrained model for training
    :param arch: vgg19,densenet121,alexnet
    :return: input size, model and drop probability
    """
    if arch == 'densenet121':
        input_size = 1024
        model = models.densenet121(pretrained=True)
        p = 0.33
    elif arch == 'vgg19':
        input_size = 25088
        model = models.vgg19(pretrained=True)
        p = 0.5
    elif arch == 'alexnet':
        input_size = 9216
        model = models.alexnet(pretrained=True)
        p = 0.5
    else:
        raise ValueError('Unexpected network architecture', arch)

    return input_size,model,p

def build_nn(model,input_size=1024,p=0.33,hidden_units=512,output_size=102):
    """
    This function is responsible for building the neural network with specified hidden units
    :param model: pretrained model
    :param input_size: input size
    :param p: drop probability
    :param hidden_units: hidden layers
    :param output_size: output size
    :return: model
    """
    #print("\n input_size",input_size)
    #print("\n p",p)
    #print("\nhidden_units",hidden_units)

    for param in model.parameters():
        param.requires_grad = False

    hidden_units = hidden_units.split(',')#hidden units are passed as a string with numbers seperated by comma: "800,600,400"
    hidden_units = [int(x) for x in hidden_units]
    hidden_units.append(output_size)

    hidden_layers = nn.ModuleList([nn.Linear(input_size,hidden_units[0])])
    hidden_layers_sizes = zip(hidden_units[:-1], hidden_units[1:])
    hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers_sizes])

    nn_layers = OrderedDict()

    #Build hidden layers for each specified hidden_unit
    for x in range(len(hidden_layers)):
        h_id = x + 1
        if x == 0:
            nn_layers.update({'drop{}'.format(h_id):nn.Dropout(p)})
            nn_layers.update({'fc{}'.format(h_id):hidden_layers[x]})
        else:
            nn_layers.update({'relu{}'.format(h_id):nn.ReLU()})
            nn_layers.update({'drop{}'.format(h_id):nn.Dropout(p)})
            nn_layers.update({'fc{}'.format(h_id):hidden_layers[x]})

    nn_layers.update({'output':nn.LogSoftmax(dim=1)})

    classifier = nn.Sequential(nn_layers)

    #Update classfier to new classifier
    model.classifier = classifier
    #print("\nclassifier",model.classifier)

    return model


# Implement a function for the validation pass
def validation(model, testloader, criterion):
    #print("\n Inside Validation Function")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    accuracy = 0
    model.to(device)
    for inputs, labels in testloader:


        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    print("\n Inside do_deep_learning Function ..... Training the model")
    epochs = epochs
    print_every = print_every
    steps = 0
    running_loss = 0
    # change to cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0

                # Make sure training is back on
            model.train()



def check_accuracy_on_test(testloader):
    correct = 0
    total = 0
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def save_model():
    # TODO: Save the checkpoint
    check_point_file = 'checkpoint.pth'
    model.class_to_idx = train_image_datasets.class_to_idx
    checkpoint_dict = {
            'arch': arch,
            'input_size':input_size,
            'output_size':102,
            'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'state_dict': model.state_dict()
            }

    torch.save(checkpoint_dict, path)
    print('Model saved at {}'.format(path))

def load_model(file='checkpoint.pth'):
    chkpt = torch.load(file,map_location='cpu')
    model.load_state_dict(chkpt['state_dict'])
    model.class_to_idx = chkpt['class_to_idx']
    model.input_size = chkpt['input_size']
    model.hidden_units = chkpt['hidden_units']
    model.output_size =chkpt['output_size']
    model.classifier = chkpt['classifier']
    return model

if __name__ == '__main__':
    args = get_commandlineArgs()
    #if '--arch' in args:
    if args.arch is not None:
        arch = args.arch
        #print('ifarch',args.arch)
    else:
        #print('elsearch')
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)
        input_size = 1024

    if args.hidden_units is not None:
        hidden_units = args.hidden_units
    else:
        hidden_units = '512'

    if args.save_dir is not None:
        path = args.save_dir + "/checkpoint.pth"

    else:
        path = 'checkpoint.pth'
        #print("epath =",path)
    #if args.epochs:
    epochs = args.epochs
    #if args.learning_rate:
    learning_rate = args.learning_rate
    #if args.gpu:
    gpu = args.gpu
    #if args.checkpoint:
    checkpoint = args.checkpoint

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU available training neural network in GPU mode ...... !!!!")
    else:
        print("GPU not available .....Training in CPU mode !!!!")

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    def print_details():
        os.system('clear')
        print("\n\nCommand Line arguments used .....\n")
        print("\nPretrained network architecture :\t",arch)
        print("\nNumber of epochs:\t",epochs)
        print("\nLearning rate used to train:\t",args.learning_rate)
        print("\nHidden Layers in neural network :\t",hidden_units)
        print("\nTraining running on (cpu/gpu)= ",device)
     

    #invoke data_transformation function
    training_data_transforms,test_validation_data_transforms = data_transformations()
    #invoke data_loaders function to define data loaders
    dataloaders_train,dataloaders_validation,dataloaders_test,train_image_datasets = data_loaders()
    #invoke load_labels function to get the labels
    cat_to_name = load_labels()
    #invoke pretrained_model function to load the pretrained model
    input_size, model,p = pretrained_model(arch)
    #invoke build_nn function to build the neural network based on specified layers
    model = build_nn(model,input_size,p,hidden_units)
    #define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    #initialize the start time before training the network to compute the total time consumed in training
    start = time.time()
    print_details()
    #invoke do_deep_learning function to train the network
    do_deep_learning(model, dataloaders_train,dataloaders_validation, epochs, 40, criterion, optimizer, gpu)
    print(f"Total Time for training: {(time.time() - start)/3:.3f} seconds")

    #invoke check_accuracy_on_test function to find the accuracy on the test dataset
    check_accuracy_on_test(dataloaders_test)

    #invoke save_model function to save the trained model as checkpoint.pth
    save_model()
    #invoke load_model function to load the saved network
    mymodel = load_model(path)
    #print(mymodel)
