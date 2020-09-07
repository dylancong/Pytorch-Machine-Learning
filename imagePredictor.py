import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, utils
from cifar import CIFAR10
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import os.path
import pickle
from settingReader import settingReader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), padding=1), #used to be 4 by 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4,4), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2048, 1024), #used to be 18432, 3072
            nn.ReLU(),
            nn.Linear(1024, 512), #used to be 3072, 1024
            nn.ReLU(),
            nn.Linear(512, 256),  #used to be 1024, 512
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 20)#usded to be 512,20
        )

    def forward(self, x):
        if torch.cuda.is_available():
           x = x.to(device="cuda")
        # conv layers
        x = self.conv_layer(x)
        # print(x.size(-4))
        # print(x.size(-3))
        # print(x.size(-2))
        # print(x.size(-1))
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        # print(x.size(3))
        # flatten
        x = x.view(x.size(0), -1)


        # fc layer
        x = self.fc_layer(x)

        return x
mod_path = Path(__file__).parent
path = (mod_path / "path.pth").resolve()


model = CNN()
model.load_state_dict(torch.load(path))
model.eval()
data = []
file_path = (mod_path / "Data" / "test.pickle")
with open(file_path, 'rb') as f:
    entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    # if 'labels' in entry:
    #     self.targets.extend(entry['labels'])
    # else:
    #     self.targets.extend(entry['fine_labels'])

data = np.vstack(data)
data = data.reshape(-1,3,32,32)
data = data.transpose((0,2,3,1))
test_batch_size = 1
transform = transforms.Compose([transforms.ToTensor()])
images = CIFAR10(root='./data', train=False,download=True, transform=transform, inputVersion = True)
val_loader = torch.utils.data.DataLoader(images, batch_size=test_batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
classNames = settingReader().getItem("classNames")
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            

            # print(target)
            output = model(data)
            _, pred = output.max(1)
            data = data.squeeze()
            data = data.permute(1,2,0)

            print(classNames[pred[0].item()-1])
            torchvision.transforms.ToPILImage()(data)

            plt.imshow(data)
            
    
test(model,val_loader)