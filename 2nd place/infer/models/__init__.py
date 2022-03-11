
from torch.cuda.amp import autocast
import torch.nn as nn
import segmentation_models_pytorch as smp
class CloudModel(nn.Module):
    def __init__(self, encoder: str, network: str, in_channels: int = 4, n_class: int = 1,
                 pre_train="imagenet", **kwargs):
        super(CloudModel, self).__init__()
        self.smp_model_name = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3",
                               "DeepLabV3Plus", "PAN"]
        self.model = getattr(smp,network)(
            encoder_name=encoder,encoder_weights=pre_train,in_channels=in_channels,classes=n_class,** kwargs
        )
    @autocast()
    def forward(self, x):
        x = self.model(x)
        return x
