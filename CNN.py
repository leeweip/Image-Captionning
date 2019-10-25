import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, embedding_size):
        super(CNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        #modules= list(self.resnet152.children())[:-1]
        #self.resnet152 = nn.Sequential(*modules)

        # output size 1000 (originally from ResNet) --> 512 (embedding_size)
        self.linear = nn.Linear(self.resnet.fc.in_features, embedding_size)
        self.resnet.fc = self.linear
        #self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, images):
        emb = self.resnet(images)
        #emb = self.linear(emb)
        #emb = emb.view(emb.size(0), -1)
        return emb

