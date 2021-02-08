import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['okddip_vgg16']


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout=0.0, number_net=4):
        super(VGG, self).__init__()
        self.inplances = 64
        self.number_net = number_net
        self.num_branches = number_net
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)

        fix_planes = self.inplances
        for i in range(self.number_net):
            self.inplances = fix_planes
            setattr(self, 'layer4_' + str(i), self._make_layers(512, num_layer))
            setattr(self, 'classifier_' + str(i), nn.Sequential(
                                                    nn.Linear(512, num_classes),
                                                ))

        self.query_weight = nn.Linear(512, 512 // 8, bias=False)
        self.key_weight = nn.Linear(512, 512 // 8, bias=False)
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
        input = self.layer3(x)

        logit = []
        x = getattr(self, 'layer4_' + str(0))(input)
        x = x.view(x.size(0), -1)

        x_3 = x
        proj_q = self.query_weight(x_3)  # B x 8
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)  # B x 8
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier_0')(x_3)  # B x num_classes
        logit.append(x_3_1)
        pro = x_3_1.unsqueeze(-1)

        for i in range(1, self.number_net - 1):
            x = getattr(self, 'layer4_' + str(i))(input)
            x = x.view(x.size(0), -1)
            temp = x
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]

            temp_1 = getattr(self, 'classifier_' + str(i))(temp)
            logit.append(temp_1)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)  # B x num_classes x num_branches
            proj_q = torch.cat([proj_q, temp_q], 1)  # B x num_branches x 8
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0, 2, 1))

        temp = getattr(self, 'layer4_' + str(self.num_branches - 1))(input)
        temp = temp.view(temp.size(0), -1)
        temp_out = getattr(self, 'classifier_' + str(self.num_branches - 1))(temp)

        return pro, x_m, temp_out


def okddip_vgg16(**kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    return model


if __name__ == '__main__':
    net = okddip_vgg16(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
