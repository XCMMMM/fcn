from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # 看一下我们传入的return_layers的key值，即layer3和layer4是否在我们的网络中，不在会报错
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        # 因为在我们的fcn结构图中，并没有使用到resnet的所有层，比如最后的全局平均池化层和全连接层
        # 而我们的模型又是以之前定义的完整的resnet结构为基础定义的，因此要将没有用到的层删掉
        # 通过for循环遍历我们定义的model中的每一层模型
        # 将module存储到layers中，然后进行判断，如果该层在return_layers中（包含layer3和layer4），就将return_layers中的该项删掉
        # 最后将layer4删掉后，return_layers就为空了，会退出循环，而此时我们的模型构建到layer4也结束了
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 构建一个有序字典
        out = OrderedDict()
        # for循环取出我们的每一个子模块module，将传入的数据x依次传入module进行一个正向传播
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                # self.return_layers['layer3']='aux'，self.return_layers['layer4']='out'
                out_name = self.return_layers[name]
                # 得到我们的输出：out['aux']和out['out']
                out[out_name] = x
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # x是我们的数据，取出后两维的维度，也就是我们的高宽
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        # 将x输入到backbone中可以得到我们的输出，输出为有序字典
        features = self.backbone(x)
        # 构建有序字典，赋值给result
        result = OrderedDict()
        # 提取出layer4的输出
        x = features["out"]
        # 将layer4的输出输入到主分类器中，得到x
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        # 将上面得到的输出x进行双线性插值，还原到图片的原始大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            # 提取出layer3的输出
            x = features["aux"]
            # 将layer3的输出输入到辅助分类器中，得到x
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            # 将上面得到的输出x进行双线性插值，还原到图片的原始大小
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            # 经过第一个卷积层后，输入特征层的channel变为原来的1/4，即从原来的1024变为256，具体可见fcn结构图
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 输出channel是num_classes
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    # replace_stride_with_dilation列表False对应于fcn结构图中的Layer2，没有修改；第二个Ture对应于Layer3；第三个True对应于Layer4
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    # 对应于layer4的输出，输出特征矩阵的channel为2048，可以通过fcn结构图看出
    out_inplanes = 2048
    # 对应于layer3的输出，输出特征矩阵的channel为1024（因为要实用辅助分类器，辅助分类器的输入就是layer3的输出）
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    # 如果使用辅助分类器的话，就要拿出我们layer3的输出
    if aux:
        return_layers['layer3'] = 'aux'
    # 通过类IntermediateLayerGetter对我们的resnet网路进行一个重构
    # 传入的参数为之前构建的resnet50网络backbone和return_layers字典
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 默认设置aux_classifier=None，即不使用辅助分类器
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    # 如果使用辅助分类器的话，就构建fcn结构图中的FCN Head，传入的aux_inplanes为layer3的输出channel
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    # 使用FCNHead类对我们主分支上的FCN Head进行构建，传入的out_inplanes为layer4的输出channel
    classifier = FCNHead(out_inplanes, num_classes)
    # 将backbone和classifier和辅助分类器aux_classifier传入类FCN构建我们的model
    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
