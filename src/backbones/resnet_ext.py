from tf2_resnets.blocks import stack1, stack2, stack3
from backbones.tf2_resnet import ResNet
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


def ResNet18(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_mixstyle=False,
    **kwargs
):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack3(x, 64, 2, stride1=1, conv_shortcut=False, name="conv2")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv2_mixstyle")(x)
        x = stack3(x, 128, 2, name="conv3")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv3_mixstyle")(x)
        x = stack3(x, 256, 2, name="conv4")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv4_mixstyle")(x)
        return stack3(x, 512, 2, name="conv5")

    return ResNet(
        stack_fn,
        False,
        "resnet18",
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


def ResNet34(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_mixstyle=False,
    **kwargs
):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack3(x, 64, 3, stride1=1, conv_shortcut=False, name="conv2")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv2_mixstyle")(x)
        x = stack3(x, 128, 4, name="conv3")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv3_mixstyle")(x)
        x = stack3(x, 256, 6, name="conv4")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv4_mixstyle")(x)
        return stack3(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        "resnet34",
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


def ResNeSt50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_mixstyle=False,
    **kwargs
):
    """Instantiates the ResNeSt50 architecture."""

    def stack_fn(x):
        x = stack2(
            x, 64, 3, stride1=1, base_width=64, radix=2, is_first=False, name="conv2"
        )
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv2_mixstyle")(x)
        x = stack2(x, 128, 4, base_width=64, radix=2, name="conv3")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv3_mixstyle")(x)
        x = stack2(x, 256, 6, base_width=64, radix=2, name="conv4")
        if use_mixstyle:
            x = MixStyle(p=0.5, alpha=0.1, name="conv4_mixstyle")(x)
        return stack2(x, 512, 3, base_width=64, radix=2, name="conv5")

    return ResNet(
        stack_fn,
        False,
        "resnest50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        True,
        32,
        classes,
        **kwargs
    )
