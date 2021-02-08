import torch
import torch.nn as nn
import math

__all__ = ['cl_ilr_vgg16']


class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output / num_branches, None


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout=0.0, number_net=4):
        super(VGG, self).__init__()
        self.inplances = 64
        self.number_net = number_net
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_ILR = ILR.apply

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)

        fix_planes = self.inplances
        for i in range(math.ceil(self.number_net / 2)):
            self.inplances = fix_planes
            setattr(self, 'layer3_' + str(i), self._make_layers(512, num_layer))

        fix_planes = self.inplances
        for i in range(self.number_net):
            self.inplances = fix_planes
            setattr(self, 'layer4_' + str(i), self._make_layers(512, num_layer))
            setattr(self, 'classifier_' + str(i), nn.Sequential(
                                                    nn.Linear(512, num_classes),
                                                ))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        logits = []

        x = self.layer_ILR(x, 2)

        x2 = []
        inputs = x
        for i in range(0, math.ceil(self.number_net / 2)):
            x = getattr(self, 'layer3_' + str(i))(inputs)
            if self.number_net - i * 2 >= 2:
                x = self.layer_ILR(x, 2)
                x2.append(x)
                x2.append(x)
            else:
                x2.append(x)

        for i in range(self.number_net):
            x = getattr(self, 'layer4_' + str(i))(x2[i])
            x = x.view(x.size(0), -1)
            x = getattr(self, 'classifier_' + str(i))(x)
            logits.append(x)
        return logits


def cl_ilr_vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


if __name__ == '__main__':
    net = cl_ilr_vgg16(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
