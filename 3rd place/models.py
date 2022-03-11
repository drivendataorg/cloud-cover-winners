import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import timm
from timm.models import create_model
from timm.models.layers import Conv2dSame


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvSilu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class Timm_Unet(nn.Module):
    def __init__(self, name='nfnet_l0', pretrained=True, inp_size=3, otp_size=1, checkpoint_path='', **kwargs):
        super(Timm_Unet, self).__init__()
        

        encoder = create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size, checkpoint_path=checkpoint_path)

        encoder_filters = [f['num_chs'] for f in encoder.feature_info]

        decoder_filters = [32, 48, 96, 128, 256]

        self.conv6 = ConvSilu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvSilu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvSilu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvSilu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvSilu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvSilu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self._initialize_weights()

        self.encoder = encoder


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1, enc2, enc3, enc4, enc5 = self.encoder(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class EfficientNet_Timm_Unet(nn.Module):
    def __init__(self, name='tf_efficientnet_b7_ns', pretrained=True, inp_size=3, checkpoint_path='', **kwargs):
        super(EfficientNet_Timm_Unet, self).__init__()
        
        enc_sizes = {
            'tf_efficientnet_b0_ns': [32, 24, 40, 112, 1280],
            'tf_efficientnet_b1_ns': [32, 24, 40, 112, 1280],
            'tf_efficientnet_b2_ns': [32, 24, 48, 120, 1408],
            'tf_efficientnet_b3_ns': [40, 32, 48, 136, 1536],
            'tf_efficientnet_b4_ns': [48, 32, 56, 160, 1792],
            'tf_efficientnet_b5_ns': [48, 40, 64, 176, 2048],
            'tf_efficientnet_b6_ns': [56, 40, 72, 200, 2304],
            'tf_efficientnet_b7_ns': [64, 48, 80, 224, 2560],
            'efficientnet-b8': [32, 56, 88, 248, 2816],
            'tf_efficientnetv2_b1': [32, 32, 48, 112, 1280],
            'tf_efficientnetv2_b0': [32, 32, 48, 112, 1280],
            'tf_efficientnetv2_s': [24, 48, 64, 160, 1280],
        }

        encoder_filters = enc_sizes[name]

        decoder_filters = np.asarray([32, 48, 96, 128, 256])


        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = create_model(name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(encoder_filters))])

        if inp_size != 3:
            _old = self.encoder_stages[0][0]
            new_conv_stem = Conv2dSame(inp_size, _old.out_channels, _old.kernel_size, _old.stride, _old.padding, _old.dilation, _old.groups, _old.bias)
            self.encoder_stages[0][0] = new_conv_stem


    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv_stem,
                encoder.bn1,
                encoder.act1
            )
        elif layer == 1:
            return nn.Sequential(encoder.blocks[:2])
        elif layer == 2:
            return nn.Sequential(encoder.blocks[2:3])
        elif layer == 3:
            return nn.Sequential(encoder.blocks[3:5])
        elif layer == 4:
            return nn.Sequential(
                *encoder.blocks[5:],
                encoder.conv_head,
                encoder.bn2,
                encoder.act2
            )


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        enc1, enc2, enc3, enc4, enc5 = enc_results

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Resnet_Timm_Unet(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, inp_size=3, checkpoint_path='', **kwargs):
        super(Resnet_Timm_Unet, self).__init__()

        enc_sizes = {
            'resnet34': [64, 64, 128, 256, 512],  # 21.80
            'resnest14d': [64, 256, 512, 1024, 2048],  # 10.61
        }

        encoder_filters = enc_sizes[name]

        decoder_filters = np.asarray([8, 16, 24, 32, 64])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = create_model(name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(encoder_filters))])

        if inp_size != 3:
            if 'resnest' in name:
                _old = self.encoder_stages[0][0][0]
                self.encoder_stages[0][0][0] = nn.Conv2d(inp_size, _old.out_channels, _old.kernel_size, stride=_old.stride,
                    padding=_old.padding, dilation=_old.dilation, groups=_old.groups, bias=_old.bias)
            else:
                _old = self.encoder_stages[0][0]
                self.encoder_stages[0][0] = nn.Conv2d(inp_size, _old.out_channels, _old.kernel_size, stride=_old.stride,
                    padding=_old.padding, dilation=_old.dilation, groups=_old.groups, bias=_old.bias)


    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.act1)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        enc1, enc2, enc3, enc4, enc5 = enc_results

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
