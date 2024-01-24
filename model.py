from torch import nn
import torch

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv8 = nn.Sequential(
             nn.Conv2d(256,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(25088, 4096),
            nn.Sigmoid(), 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 100),
        )
    def forward_once(self,x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.conv10(output)
        output = self.conv11(output)
        output = self.conv12(output)
        output = self.conv13(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
    
    def forward(self,input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2