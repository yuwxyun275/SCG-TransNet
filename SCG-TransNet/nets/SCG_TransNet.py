import copy
import importlib

import torch
from torch import nn

from model.encoder import build_encoder
from model.decoder import build_decoder, BasicLayer_up, SwinDecoder
from model.aspp import build_aspp, build_convaspp
import torch.nn.functional as F
from attention import se_block, cbam_block, eca_block, PAM_CAM_Layer, Att
attention_block = [se_block, cbam_block, eca_block, PAM_CAM_Layer, Att]

class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, aspp_config, decoder_config):
        super().__init__()
        self.encoder = build_encoder(encoder_config)
        self.aspp = build_aspp(input_size=self.encoder.high_level_size,
                               input_dim=self.encoder.high_level_dim,
                               out_dim=self.encoder.low_level_dim, config=aspp_config)

        self.convaspp = build_convaspp(dim_in = 384, dim_out= 384)
        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_dim=self.encoder.low_level_dim,
                                     config=decoder_config)

        self.shallow_feature_conv = nn.Sequential(
            nn.Conv2d(192, 96, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.deep_feature_conv = nn.Sequential(
            nn.Conv2d(384, 96, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.att_right = attention_block[4](384)
        self.att_down = attention_block[4](96)

    def run_encoder(self, x):
        low_level, high_level, down_x1 = self.encoder(x)
        return low_level, high_level, down_x1

    def run_aspp(self, x, x_convaspp):
        return self.aspp(x, x_convaspp)

    def run_convaspp(self,x):
        return self.convaspp(x)

    def run_decoder(self, low_level, pyramid, fpn_x):
        return self.decoder(low_level, pyramid, fpn_x)

    def run_upsample(self, x):
        return self.upsample(x)


    def forward(self, x):
        low_level, high_level, down_x1 = self.run_encoder(x) #down_x1

        low_level = low_level.view(low_level.shape[0], low_level.shape[3], low_level.shape[1], low_level.shape[2])
        low_level = self.att_down(low_level)
        low_level = low_level.view(low_level.shape[0], low_level.shape[2], low_level.shape[3], low_level.shape[1])

        high_level = high_level.view(high_level.shape[0],high_level.shape[3],high_level.shape[1],high_level.shape[2])
        down_x2 = self.deep_feature_conv(high_level)
        high_level = high_level.view(high_level.shape[0], high_level.shape[2], high_level.shape[3], high_level.shape[1])
        down_x2 = F.interpolate(down_x2, size=(32,32), mode='bilinear',
                          align_corners=True)


        down_x1 = down_x1.view(down_x1.shape[0],down_x1.shape[3],down_x1.shape[1],down_x1.shape[2])
        down_x1 = self.shallow_feature_conv(down_x1)
        fpn_x = down_x1 + down_x2
        fpn_x = fpn_x.view(fpn_x.shape[0],fpn_x.shape[2],fpn_x.shape[3],fpn_x.shape[1]) #torch.Size([1, 32, 32, 96])
        high_level = high_level.view(high_level.shape[0], high_level.shape[3], high_level.shape[1], high_level.shape[2])
        high_level = self.att_right(high_level)
        high_level = high_level.view(high_level.shape[0], high_level.shape[2], high_level.shape[3], high_level.shape[1])

################################SCD#######################################
        x_convaspp = self.run_convaspp(high_level.permute(0, 3, 1, 2))
        x_convaspp = x_convaspp.permute(0,2,3,1)

        x = self.run_aspp(high_level, x_convaspp)
###########################################################################
        x = self.run_decoder(low_level, x, fpn_x)

        return x


if __name__ == '__main__':
    model_config = importlib.import_module(f'model.configs.swin_224_7_4level')
    net = SwinDeepLab(
        model_config.EncoderConfig,
        model_config.ASPPConfig,
        model_config.DecoderConfig
    )
    x = torch.rand(4, 3, 256, 256)
    y = net(x)
    print(y.shape)
