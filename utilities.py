import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import random
import os
import argparse

#Define a function to get extract and store raw input from command line
def get_input_args():
    parser = argparse.ArgumentParser()
    #Creates command line arguments  
    parser.add_argument('--save', type = bool, default = True, help = 'opt whether to save model after running')
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'desired CNN model architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'desired learning rate')
    parser.add_argument('--hidden_layers', type = int, default = 256, help = 'desired hidden layers')
    parser.add_argument('--epochs', type = int, default = 5, help = 'desired number of epochs')
    parser.add_argument('--GPU', type = str, default = 'Y', help = 'Use GPU? Y/N')
    parser.add_argument('--TopK', type = int, default = 5, help = 'Top K predictions')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'JSON file with dic mapping of categories to names')
    return parser.parse_args()

def transformation(directory,train=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    #defining the transformations for training and testing data
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(45),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       normalize])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

    #Loading the datasets with ImageFolder and defining the dataloaders
    
    if train:
        data = datasets.ImageFolder(directory, transform=train_transforms)
        dataloader = torch.utils.data.DataLoader(data, batch_size=34, shuffle=True)
        
    else:
        data = datasets.ImageFolder(directory, transform=test_transforms)
        dataloader = torch.utils.data.DataLoader(data, batch_size=34)
        
    return dataloader, data


def process_image(image):
    #Setting size parameters for image resizing
    size = 256, 256
    #Open the image
    pil_image = Image.open(image)
    #display(pil_image)
    #Resizing the image while maintaining the aspect ratio
    pil_image.thumbnail(size, Image.ANTIALIAS)
    #Extracting width and height parameters from the image
    width, height = pil_image.size
    #Preparing to center crop the image
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    #Center cropping the image
    pil_image = pil_image.crop((left, top, right, bottom))
    #Converting image to numpy array, squeezing and normalizing the array
    np_image = np.array(pil_image)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (np_image - mean) / std
    #Transposing the color channel to the first dimension of the array
    img = img.transpose((2, 0, 1))
    #Converting numpy array to tensor
    y = torch.from_numpy(img)
    return y


def imshow(image, ax=None, title=None, axes=True):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    if not axes:
    
        ax.axis('off')
    
    return ax


def predict(image_path, model, topk=5):
    #define device and load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #set model to evaluation mode
    model.eval()
    #convert image to a tensor and adjust tensor dimensions for input into the model
    image = process_image(image_path).float()
    img = image.unsqueeze(0)
    #grab the class to index attribute from our model
    class_to_idx = model.class_to_idx
    #inverting the dictionary attribute from the model
    inv_class_to_idx = {v: k for k, v in class_to_idx.items()}
    #save memory by instructing not to backpropagate
    with torch.no_grad():
        img = img.to(device)
        output = model(img)
        #find top k predictions for image
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        y = top_class.cpu().numpy()
        z = []
        for i in y:
            for j in i:
                #provides the class label based on the index output
                z.append(inv_class_to_idx[j])
        #returns the probabilites and classes as lists
        return top_p.cpu().numpy().tolist()[0],z

