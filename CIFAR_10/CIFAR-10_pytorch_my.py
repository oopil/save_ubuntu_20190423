"""
reference :
https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""

import torch
import torchvision
import torchvision.transforms as transforms

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
#%%
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
# import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#%%

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        last_image_size = 32
        last_size_bottle = last_image_size
        super(Net, self).__init__()

        # batch normarlization
        self.batchnorm = nn.BatchNorm2d(3).cuda()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1).cuda()
        # weight - xavier initialization
        nn.init.xavier_normal(self.conv1.weight).cuda()
        self.pool = nn.MaxPool2d(2, 2).cuda()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1).cuda()
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1).cuda()
        self.conv_128_half = nn.Conv2d( last_image_size,last_size_bottle, 1, stride=1, padding=0).cuda()
        self.conv_64_twice = nn.Conv2d(last_size_bottle, last_image_size, 1, stride=1, padding=0).cuda()
        self.conv_64_same = nn.Conv2d(last_size_bottle, last_size_bottle, 3, stride=1, padding=1).cuda()

        # xavier normal initialize
        nn.init.xavier_normal(self.conv2.weight).cuda()
        self.fc1 = nn.Linear(last_image_size * 8 * 8, 512).cuda()
        self.fc2 = nn.Linear(512, 256).cuda()
        self.fc3 = nn.Linear(256, 10).cuda()

        #self.dropout = nn.Dropout(0.8)

    def bottle_neck(self, x, ch):
        x = F.relu(self.conv_128_half(x))
        x = F.relu(self.conv_64_same(x))
        x = F.relu(self.conv_64_twice(x))
        #x = F.relu(nn.Conv2d(ch, ch/2, 1, stride=1, padding=0))
        #x = F.relu(nn.Conv2d(ch/2, ch/2, 3, stride=1, padding=1))
        #x = F.relu(nn.Conv2d(ch/2, ch, 1, stride=1, padding=0))
        return x

    def res_net(self, x, width_num, ch):
        output = x
        for count in range(1, width_num+1):
            output = output + self.bottle_neck(x,ch)
        return output

    def forward(self, x):
        #x = self.batchnorm(x)
        #if torch.cuda.is_available():
        #    x = x.cuda()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv3(x))
        #assert False

        # parallel residual network part start
        #x = self.batchnorm(self.res_net(x, 3, 128))
        last_image_size = 32
        width_num = 2
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        x = self.res_net(x, width_num, last_image_size)
        # flatten x
        x = x.view(-1, last_image_size * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc3(x)
        return x


"""
#pytorch gpu test
x = torch.randn(1)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

assert False
"""

model = Net()
print(model)
#assert False
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0.01, weight_decay=0.9)

correct = 0
total = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            device = torch.device("cuda")  # a CUDA device object
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = torch.ones_like(labels, device=device)
        # get the inputs
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        with torch.no_grad():
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f *** Accuracy: %d %%' %(epoch + 1, i + 1, running_loss / 2000, 100 * correct / total))
                running_loss = 0.0
print('Finished Training')

# model save and restore if you want
print('save pytorch model')
torch.save(model, '/tmp/cifar10-model_resnet.pt')

restore = False
if restore:
    model = torch.load('/tmp/cifar10-model_resnet.pt')
#%%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
