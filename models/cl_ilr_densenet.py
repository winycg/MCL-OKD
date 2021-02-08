import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

__all__ = ['cl_ilr_densenetd40k12']


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output / num_branches, None


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, number_net=4, block_config=(16, 16, 16), num_branches=3, bpscale=False, avg=False,
                 compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, ind=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.num_branches = number_net
        self.avg = avg
        self.ind = ind
        self.bpscale = bpscale
        self.layer_ILR = ILR.apply
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

            elif i == len(block_config) - 2:
                for i in range(self.num_branches-2):
                    setattr(self, 'layer2_' + str(i), _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                ))
                num_features = num_features + num_layers * growth_rate
                for i in range(self.num_branches):
                    setattr(self, 'layer2_tran' + str(i), _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression)))
                num_features = int(num_features * compression)
            else:
                for i in range(self.num_branches):
                    setattr(self, 'layer3_' + str(i), _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                ))

        num_features = num_features + num_layers * growth_rate
        for i in range(self.num_branches):
            setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
            setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace=True))
        # Linear layer
        for i in range(self.num_branches):
            setattr(self, 'classifier_' + str(i), nn.Linear(num_features, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.features(input)
        x = self.layer_ILR(x, 2)

        logits = []
        embeddings = []
        x1 = getattr(self, 'layer2_0')(x)
        x1 = getattr(self, 'layer2_tran0')(x1)

        x2 = getattr(self, 'layer2_1')(x)
        x2 = getattr(self, 'layer2_tran1')(x2)

        x1 = self.layer_ILR(x1, 2)
        x2 = self.layer_ILR(x2, 2)

        x = getattr(self, 'layer3_0')(x1)
        x = getattr(self, 'norm_final_0')(x)
        x = getattr(self, 'relu_final_0')(x)
        x = self.avgpool(x)
        embeddings.append(x)
        x = x.view(x.size(0), -1)
        x = getattr(self, 'classifier_0')(x)
        logits.append(x)

        x = getattr(self, 'layer3_1')(x1)
        x = getattr(self, 'norm_final_1')(x)
        x = getattr(self, 'relu_final_1')(x)
        x = self.avgpool(x)
        embeddings.append(x)
        x = x.view(x.size(0), -1)
        x = getattr(self, 'classifier_1')(x)
        logits.append(x)

        x = getattr(self, 'layer3_2')(x2)
        x = getattr(self, 'norm_final_2')(x)
        x = getattr(self, 'relu_final_2')(x)
        x = self.avgpool(x)
        embeddings.append(x)
        x = x.view(x.size(0), -1)
        x = getattr(self, 'classifier_2')(x)
        logits.append(x)

        x = getattr(self, 'layer3_3')(x2)
        x = getattr(self, 'norm_final_3')(x)
        x = getattr(self, 'relu_final_3')(x)
        x = self.avgpool(x)
        embeddings.append(x)
        x = x.view(x.size(0), -1)
        x = getattr(self, 'classifier_3')(x)
        logits.append(x)

        return logits


def cl_ilr_densenetd40k12(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD40K12 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = DenseNet(growth_rate=12, block_config=[6, 6, 6], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


if __name__ == '__main__':
    net = cl_ilr_densenetd40k12(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
