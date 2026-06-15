import segmentation_models_pytorch as smp

def build_unet(effnet_version="efficientnet-b0", n_classes=4, pretrained=True):
    model = smp.Unet(
        encoder_name=effnet_version,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=n_classes
    )
    return model
