import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from utilities import get_input_args, transformation
from model import buildmodel
from workspace_utils import active_session

def main():
    
    in_arg = get_input_args()
    
    data_dir = 'flowers'
    
    train_dir = data_dir + '/train' 
    valid_dir = data_dir + '/valid'
    
    trainloader, train_data = transformation(train_dir,train=True)
    validloader, valid_data = transformation(valid_dir,train=False)

    #Extract inputs from argparser
    model_type = in_arg.arch
    hl = in_arg.hidden_layers
    learn = in_arg.learning_rate
    
    # Use GPU if it's available
    device = torch.device("cuda" if in_arg.GPU == 'Y' else "cpu")
    
    #build model based on inputs
    model, criterion, optimizer = buildmodel(model_type,hl,learn)

    model.to(device);

    #Training model
    
    #Setting parameters to train network
    epochs = in_arg.epochs

    with active_session():

        for e in range(epochs):

            #reset training loss, testing loss and accuracy
            train_loss = 0
            test_loss = 0
            accuracy = 0

            #Training network
            for images, labels in trainloader:
                #place model inputs to the device
                images, labels = images.to(device), labels.to(device)
                #Set model to training mode to enable dropout
                model.train()
                #Set gradients to zero
                optimizer.zero_grad()
                #Forward pass through model
                output = model(images)
                #calculate loss
                loss = criterion(output,labels)
                #backpropagate in order to determine changes to the weights
                loss.backward()
                #update weights
                optimizer.step()
                #Keep track of training losses
                train_loss += loss.item()

            #validation step
            else:
                #set model to evaluation mode
                model.eval()
                #save memory by instructing not to backpropagate
                with torch.no_grad():
                    for images, labels in validloader:
                        #place model inputs to the device
                        images, labels = images.to(device), labels.to(device)
                        #Forward pass through model
                        output = model(images)
                        loss = criterion(output,labels)
                        test_loss += loss.item()
                        #find top prediction for each image
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        #determine if top prediction is equal to the labels
                        equals = top_class == labels.view(*top_class.shape)
                        #average over all 102 outputs
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss/len(trainloader)),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
    if in_arg.save == True:
        
        checkpoint = {'input_size': 1024,
                      'output_size': 102,
                      'hidden_layer' : hl,
                      'class_to_idx' : train_data.class_to_idx,
                      'optimizer.state_dict': optimizer.state_dict,
                      'epochs': epochs,
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, 'checkpoint.pth')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
