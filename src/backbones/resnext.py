from tf2_resnets.blocks import stack1
# from tf2_resnets.resnet import ResNet
from backbones.resnet import ResNet
from backbones.mixstyle import MixStyle


def ResNeXt50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_mixstyle=False,
    **kwargs
):
    """Instantiates the ResNeXt50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, groups=32, base_width=4, name="conv2")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv2_mixstyle")(x)
        x = stack1(x, 128, 4, groups=32, base_width=4, name="conv3")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv3_mixstyle")(x)
        x = stack1(x, 256, 6, groups=32, base_width=4, name="conv4")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv4_mixstyle")(x)
        return stack1(x, 512, 3, groups=32, base_width=4, name="conv5")

    return ResNet(
        stack_fn,
        False,
        "resnext50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        False,
        None,
        classes,
        **kwargs
    )


def ResNeXt101(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_mixstyle=False,
    **kwargs
):
    """Instantiates the ResNeXt101 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, groups=32, base_width=8, name="conv2")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv2_mixstyle")(x)
        x = stack1(x, 128, 4, groups=32, base_width=8, name="conv3")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv3_mixstyle")(x)
        x = stack1(x, 256, 23, groups=32, base_width=8, name="conv4")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv4_mixstyle")(x)
        return stack1(x, 512, 3, groups=32, base_width=8, name="conv5")

    return ResNet(
        stack_fn,
        False,
        "resnext101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        False,
        None,
        classes,
        **kwargs
    )
