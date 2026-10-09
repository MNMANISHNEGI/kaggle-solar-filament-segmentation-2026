"""U-Net with an ImageNet-pretrained encoder and scSE attention in the decoder."""

import segmentation_models_pytorch as smp
import torch


def build_model(encoder="resnet50", pretrained=True, in_channels=3):
    weights = "imagenet" if pretrained else None
    try:
        return smp.Unet(encoder, encoder_weights=weights, in_channels=in_channels, classes=1,
                        decoder_attention_type="scse")
    except Exception as e:  # e.g. no internet access to download weights
        if weights is None:
            raise
        print(f"Pretrained weights unavailable ({e}); building {encoder} from scratch")
        return smp.Unet(encoder, encoder_weights=None, in_channels=in_channels, classes=1,
                        decoder_attention_type="scse")


def load_model(checkpoint, encoder="resnet50", device="cpu"):
    """Build the architecture without downloading weights and load a trained checkpoint."""
    model = build_model(encoder, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()
