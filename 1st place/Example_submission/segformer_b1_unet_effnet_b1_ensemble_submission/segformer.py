import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp 
from transformers import SegformerForSemanticSegmentation

class SegFormer_b1(nn.Module):
    def __init__(self):
        super(SegFormer_b1, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-ade-512-512')
        self.segformer.decode_head.classifier = nn.Conv2d(256,1,kernel_size=1)
    # @torch.cuda.amp.autocast()
    def forward(self, image):
        image = image[:,0:3]
        
        batch_size = len(image)
        with amp.autocast():
            mask = self.segformer(image).logits
            mask = F.interpolate(mask, image.shape[-2:], mode="bilinear", align_corners=True)
            
        return mask
    

class AmpNet(SegFormer_b1):
    
    def __init__(self):
        super(AmpNet, self).__init__()
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)

  #True #False
