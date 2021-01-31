from backbones.densenet import DenseNet121, DenseNet169, DenseNet201
from backbones.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
)
from backbones.inception_resnet_v2 import InceptionResNetV2
from backbones.inceptionv3 import InceptionV3
from backbones.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
# from backbones.resnext import ResNeXt50
from backbones.xception import Xception


class ModelFactory(object):
    def get_model_by_name(self, name="densenet121", use_mixstyle=True):
        # densenet
        if name == "densenet121":
            return DenseNet121(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "densenet169":
            return DenseNet169(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "densenet201":
            return DenseNet201(include_top=False, use_mixstyle=use_mixstyle)

        # efficientnet
        elif name == "efficientnetb0":
            return EfficientNetB0(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "efficientnetb1":
            return EfficientNetB1(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "efficientnetb2":
            return EfficientNetB2(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "efficientnetb3":
            return EfficientNetB3(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "efficientnetb4":
            return EfficientNetB4(include_top=False, use_mixstyle=use_mixstyle)

        elif name == "inception_resnetv2":
            return InceptionResNetV2(
                include_top=False, use_mixstyle=use_mixstyle
            )
        elif name == "inceptionv3":
            return InceptionV3(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "xception":
            return Xception(include_top=False, use_mixstyle=use_mixstyle)

        # Resnet
        elif name == "resnet50":
            return ResNet50V2(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "resnet101":
            return ResNet101V2(include_top=False, use_mixstyle=use_mixstyle)
        elif name == "resnet151":
            return ResNet152V2(include_top=False, use_mixstyle=use_mixstyle)
        else:
            raise NotImplementedError
        # Resnext
        # self.resnext50 = ResNeXt50(include_top=False, weights="swsl", use_mixstyle=True)
