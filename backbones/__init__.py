from tensorflow.python.keras.engine.training import Model

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
from backbones.resnext import ResNeXt50
from backbones.xception import Xception


class ModelFactory(object):
    def __init__(self):
        # densenet
        self.densenet121 = DenseNet121(include_top=False, use_mixstyle=True)
        self.densenet169 = DenseNet169(include_top=False, use_mixstyle=True)
        self.densenet201 = DenseNet201(include_top=False, use_mixstyle=True)

        # efficientnet
        self.efficientnetb0 = EfficientNetB0(include_top=False, use_mixstyle=True)
        self.efficientnetb1 = EfficientNetB1(include_top=False, use_mixstyle=True)
        self.efficientnetb2 = EfficientNetB2(include_top=False, use_mixstyle=True)
        self.efficientnetb3 = EfficientNetB3(include_top=False, use_mixstyle=True)
        self.efficientnetb4 = EfficientNetB4(include_top=False, use_mixstyle=True)

        # inception resnetv2
        self.inception_resnetv2 = InceptionResNetV2(
            include_top=False, use_mixstyle=True
        )
        self.inceptionv3 = InceptionV3(include_top=False, use_mixstyle=True)

        # Resnet
        self.resnet50v2 = ResNet50V2(include_top=False, use_mixstyle=True)
        self.resnet101v2 = ResNet101V2(include_top=False, use_mixstyle=True)
        self.resnet152v2 = ResNet152V2(include_top=False, use_mixstyle=True)

        # Resnext
        self.resnext50 = ResNeXt50(include_top=False, weights="swsl", use_mixstyle=True)

        # Xception
        self.xception = Xception(include_top=False, use_mixstyle=True)

    def get_model_by_name(self, name="densenet121"):
        return getattr(self, name)
