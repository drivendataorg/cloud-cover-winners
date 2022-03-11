from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss

bce = SoftBCEWithLogitsLoss()
dice = DiceLoss(mode='binary')
XEDiceLoss=L.JointLoss(bce, dice, 0.5, 0.5)