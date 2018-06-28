import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(1,32,5,padding=2)   
        self.conv2=nn.Conv2d(32,64,5,padding=2)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.conv4=nn.Conv2d(128,256,3,padding=1)
        self.conv5=nn.Conv2d(256,512,3,padding=1)
        self.maxpool1=nn.MaxPool2d(2)
        self.maxpool2=nn.MaxPool2d(2)
        self.maxpool3=nn.MaxPool2d(2)
        self.maxpool4=nn.MaxPool2d(2)
        self.maxpool5=nn.MaxPool2d(2)
        self.batchnorm1=nn.BatchNorm2d(32)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.batchnorm3=nn.BatchNorm2d(128)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.batchnorm5=nn.BatchNorm2d(512)
        self.globalavg=nn.AdaptiveAvgPool2d((1,1))
        self.linear1=nn.Linear(512,1024)
        self.linear2=nn.Linear(1024,136)

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
        x=F.relu(self.conv5(x))
        x=self.maxpool5(x)
        x=self.batchnorm5(x)
        x=self.globalavg(x)
        x=x.view(x.size(0),-1)
        x=self.linear1(x)
        x=self.linear2(x)
        return x







