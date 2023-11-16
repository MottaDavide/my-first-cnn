#%%
import torch
import torchvision
import torchvision.transforms as transforms

print('si parte')

transform = transforms.ToTensor()
trainset = torchvision.datasets.FashionMNIST(root="./data",
                                      train = True,
                                      download=True,
                                      transform=transform)

train_iter = iter(trainset)
image, label = next(train_iter)
torch.min(image).item(), torch.max(image).item() #(0.0, 1.0)

# splitting train in train and validation set
train, val = torch.utils.data.random_split(trainset, [50000, 10000])


# batch size = numbers of samples per batch
batch_size = 8 # that means we will have 50000/8 batches to iterate over

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                           shuffle=False, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F

# %%
class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=256, kernel_size=3) #256 3x3 filters with no padding and no stride
        self.pool1 = nn.MaxPool2d(2, 2)  #windows is 2x2

        self.conv2 = nn.Conv2d(in_channels=256,  out_channels=512, kernel_size=3) #512 3x3 filters with no padding and no stride
        self.pool2 = nn.MaxPool2d(2, 2)  #windows is 2x2

        self.conv3 = nn.Conv2d(in_channels=512,  out_channels=1024, kernel_size=2) #512 3x3 filters with no padding and no stride
        self.pool3  = nn.MaxPool2d(2, 2)  #windows is 2x2

        # images are gettin smaller, from 28x28 to 2x2 but numbers of channels went from 1 to 1024
        
        self.flatten = nn.Flatten() #flat the images to a loooong vector for fully connected layers of 4096 elements
        
        self.fc1 = nn.Linear(in_features=4096, out_features=1024) #linear layer  -> each of the 1024 numbers are between -inf to +inf
        self.drop1 = nn.Dropout(p=0.3) #prevents overfitting, each of the above 1024 neurons has a probability of p to turn weight = 0

        self.fc2 = nn.Linear(in_features=1024, out_features=1024) #linear layer
        self.drop2 = nn.Dropout(p=0.3)

        self.out = nn.Linear(in_features=1024, out_features=10) #10 like the number of classes, 
        #be careful about the linear layer here (it should be SoftMax?) but for us the best guess is the bisggest number, so there's no need to apply SoftMax
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)


        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x)) #after that the x has shape [8 = batch_size, 1024]
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.out(x) #after that the x has shape [8 = batch_size, 10]
        

        return x
    

# set cpu or gpu 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device', device)
net = NN()
net.to(device) # move the net to gpu is available, cpu otherwise

"""
# for loop in batch size
for i, data in enumerate(trainloader):   # data = contains batch images and batch labels
    inputs, labels = data[0].to(device), data[1].to(device)  # move to gpu or cpu
    print('start', inputs.shape)
    print('after net', net(inputs).shape)
    break
"""

# num parameters
num_params = 0
for x in net.parameters():
    num_params += len(torch.flatten(x))
print(f"number of parameters: {num_params: }")

#%%
# optimization criteria
import torch.optim as optim

criterion = nn.CrossEntropyLoss() #we are doing a multiclass classification -> the loss function is the cross entropy
optimizer = optim.Adam(net.parameters(), lr=0.0001) #pass the parameters and the learning rate to ADAM optimizer (a version of SGD)


# we have to define the train loop (longer than tensorflow but more custom)

## function to train during one epoch
def train_one_epoch():
    net.train(True) # set the net to be in train mode

    running_loss = 0.0
    running_accuracy = 0.0

    for batch, data in enumerate(trainloader): #iterate over all the batches during one epoch
        
        inputs, labels = data[0].to(device), data[1].to(device)

        # backpropagation
        optimizer.zero_grad() #reset the gradients before backpropagation (did not understand)

        outputs = net(inputs) # the shape is [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item() #numbers of correct predictions
        running_accuracy += correct/ batch_size

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward() # go through the loss tensor to calculate the gradient
        optimizer.step()


        if (batch) % 500 == 499:
            avg_loss_across_batches = running_loss / 500
            avg_acc_across_batches = (running_accuracy / 500)*100
            print (f'Batch {batch + 1}, Loss {avg_loss_across_batches: .2f}, Acc {avg_acc_across_batches :.1f}%')

            running_loss = .0
            running_accuracy =.0
    print()


## function to validate during one epoch
def validate_one_epoch():
    net.train(False) # set the net to NOT be in train mode

    running_loss = 0.0
    running_accuracy = 0.0

    for batch, data in enumerate(valloader): #iterate over all the batches during one epoch
        
        inputs, labels = data[0].to(device), data[1].to(device)


        with torch.no_grad(): #dont worry about calculating gradient, we do not need the gradient
            outputs = net(inputs) # the shape is [batch_size, 10]
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item() #numbers of correct predictions
            running_accuracy += correct/ batch_size
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        _,  predicted = torch.max(outputs.data, 1)

    avg_loss_across_batches = running_loss / len(valloader)
    avg_acc_across_batches = (running_accuracy / len(valloader))*100


    print (f'Val Loss {avg_loss_across_batches: .2f}, Val Acc {avg_acc_across_batches :.1f}%')

    print('*'*20)
    print()

num_epochs = 20

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('epoch:', epoch)

    train_one_epoch()
    validate_one_epoch()

print('finish')

# COMMAND ----------
