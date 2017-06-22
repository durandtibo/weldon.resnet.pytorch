import torch
import torch.nn as nn
import torchvision.models as models

from weldon import WeldonPool2d


class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes, pooling, dense=False):
        super(ResNetWSL, self).__init__()

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet18_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet18(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet34_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet34(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet50_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet50(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet101_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet101(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)


def resnet152_weldon(num_classes, pretrained=True, kmax=1, kmin=None):
    model = models.resnet152(pretrained)
    pooling = WeldonPool2d(kmax, kmin)
    return ResNetWSL(model, num_classes, pooling=pooling)



def demo():

    # load pretrained model
    model = resnet50_weldon(num_classes=20, pretrained=True)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    # define input
    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224))
    target = torch.autograd.Variable(torch.ones(1)).long()

    # compute output
    output = model.forward(input)
    print(output)

    # backward
    loss = criterion(output, target)
    loss.backward()

    # compute dense map
    model.dense = True
    output = model.forward(input)
    print('dense output', output)

if __name__ == '__main__':
    demo()
