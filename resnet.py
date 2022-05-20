import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from avalanche.models import DynamicModule
from avalanche.benchmarks.utils import AvalancheDataset


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class NewIncrementalClassifier(DynamicModule):
    def __init__(self, in_features, initial_out_features=2, bias=True, norm_weights=False):
        """ Output layer that incrementally adds units whenever new classes are
        encountered.

        Typically used in class-incremental benchmarks where the number of
        classes grows over time.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.bias = bias
        self.classifier = torch.nn.Linear(in_features, initial_out_features, bias)
        self.norm_weights = norm_weights
        if self.norm_weights:
            self.classifier.weight.data = F.normalize(self.classifier.weight.data)

    @torch.no_grad()
    def adaptation(self, dataset: AvalancheDataset):
        """ If `dataset` contains unseen classes the classifier is expanded.

        :param dataset: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(self.classifier.out_features,
                           max(dataset.targets) + 1)

        if old_nclasses == new_nclasses:
            print(f"Classifier shape: {self.classifier.weight.shape}")
            return

        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses, self.bias)
        self.classifier.weight[:old_nclasses] = old_w

        if self.bias:
            self.classifier.bias[:old_nclasses] = old_b

        print(f"Classifier shape: {self.classifier.weight.shape}")

    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        if self.norm_weights:
            self.classifier.weight.data = F.normalize(self.classifier.weight.data)
        return self.classifier(x)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, initial_num_classes, bias_classifier, norm_weights_classifier, channels=3):
        """ Constructor
        Args:
          :param depth: number of layers.
          :param initial_num_classes: initial number of classes
          :param bias_classifier: whether use bias of the classifier
          :param norm_weights_classifier: whether normalize the classifier weights
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.initial_num_classes = initial_num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.inc_classifier = NewIncrementalClassifier(self.out_dim, initial_num_classes, bias_classifier,
                                                       norm_weights_classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.inc_classifier(x)

        return y, x


def resnet32(bias_classifier=False, norm_weights_classifier=True, num_classes=100):
    """Constructs a ResNet-32 model for CIFAR-100 (by default)
    Args:
      :param num_classes: number of classes
      :param norm_weights_classifier: whether normalize the classifier weights
      :param bias_classifier: whether use bias of the classifier
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes, bias_classifier, norm_weights_classifier)
    return model


def test():
    net = resnet32(False, True, 2)
    x = torch.rand(10, 3, 32, 32)
    y, x = net(x)
    print(y.size())
    print(x.size())


if __name__ == "__main__":
    test()
