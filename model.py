import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

densenet121 = models.densenet121(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'densenet': [densenet121, 1024], 'vgg': [vgg16, 25088]}

def buildmodel(model_type,hl,learn):
    
    model = models[model_type][0]
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
        
    #adding extra hidden layers for vgg model
    hl_1 = 10000
    hl_2 = 1000
   
    p = models[model_type][1]
    #Build model archtecture of hidden layer and final output layer. 
    #Rectified Linear Unit as activation function of hidden layer, LogSoftmax for output unit
    
    model.classifier = nn.Sequential(nn.Linear(p, hl),
                                     nn.ReLU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(hl, 102),
                                     nn.LogSoftmax(dim=1))

    #NLLLoss as corresponding cost for Softmax output
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn)
    
    return model, criterion, optimizer

def load_checkpoint(model,filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model