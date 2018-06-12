import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(1,32,5,padding=2)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.conv4=nn.Conv2d(128,256,3,padding=1)
        self.maxpool1=nn.MaxPool2d(2)
        self.maxpool2=nn.MaxPool2d(2)
        self.maxpool3=nn.MaxPool2d(2)
        self.maxpool4=nn.MaxPool2d(2)
        self.batchnorm1=nn.BatchNorm2d(32)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.batchnorm3=nn.BatchNorm2d(128)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.globalavg=nn.AdaptiveMaxPool2d((1,1))
        self.linear=nn.Linear(256,136)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.maxpool1(x)
        x=self.batchnorm1(x)
        x=F.relu(self.conv2(x))
        x=self.maxpool2(x)
        x=self.batchnorm2(x)
        x=F.relu(self.conv3(x))
        x=self.maxpool3(x)
        x=self.batchnorm3(x)
        x=F.relu(self.conv4(x))
        x=self.maxpool4(x)
        x=self.batchnorm4(x)
        x=self.globalavg(x)
        x=x.view(x.size(0),-1)
        x=F.tanh(self.linear(x))
        return x







