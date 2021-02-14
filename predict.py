import torch
from torchvision import datasets, transforms, models
from utilities import get_input_args
from model import buildmodel, load_checkpoint
import matplotlib.pyplot as plt
import json
from utilities import process_image, imshow, predict
from PIL import Image
import numpy as np
import random
import os

def main():
    
    in_arg = get_input_args()
    
    model_type = in_arg.arch
    hl = in_arg.hidden_layers
    learn = in_arg.learning_rate
    TopK = in_arg.TopK
    category_names = in_arg.category_names
    
    model, criterion, optimizer = buildmodel(model_type,hl,learn) 
    
    #Loading our model from checkpoint
    model = load_checkpoint(model, 'checkpoint.pth')
    
    data_dir = 'flowers'
    
    test_dir = data_dir + '/test' 
  
    #Generate a random image from the test directory 
    n = random.randint(1,102) 
    path = test_dir + '/{}'.format(n)
    files = os.listdir(path)
    img = random.choice(files)
    fp = path + '/{}'.format(img)
    #Processing our image to convert to tensor format
    y = process_image(fp)
    #Generating our predictions
    probs, classes = predict(fp,model,TopK)
    max_value = max(probs)
    max_index = probs.index(max_value)
    max_class_idx = classes[max_index]
    #Generating the actual flower names from the classes
    with open(category_names , 'r') as f:
        cat_to_name = json.load(f)
        title = cat_to_name[str(n)]
        max_class = cat_to_name[str(max_class_idx)]
        for i in range(len(classes)):
            classes[i] = cat_to_name[classes[i]]

    print("\nFlower Name: {}".format(title), "\n\nTop Prediction with a percentage of {:.00%}: {}".format(max_value,               max_class))
    
    if TopK > 1:
        print("\nTop {} Predictions:".format(TopK))
        for i, j in zip(classes, probs):
            print("Flower Prediction: {} - ".format(i),
                  "Probability: {:.00%}".format(j))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
