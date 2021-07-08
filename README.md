# Image-Classifier
Image Classifier in Pytorch using pre-trained neural networks.

This project involves reading in a folder of various flower images in order to train a pre-trained neural network from Torchvision. The goal
is to achieve a prediction accuracy exceeding 70%. I have included the python files as well as an ipynb file to jupyter notebooks. The order
of the command line scripts is to run train.py followed by predict.py (utilities and model are auxilliary scripts). The command line application
has the option of either the Densenet121 or the VGG16 models. The jupyter notebook only draws on the Densenet121 model.

The set up of the folders with the images is the following: there are three folders with the path 'user/flowers' followed by '/train', '/test', or '/valid'.
Each of these folders contains 102 subfolders labeled numerically (i.e. from 1-102; see JSON file for mapping to flower names) with various jpg images of a 
certain flower type within each folder.
