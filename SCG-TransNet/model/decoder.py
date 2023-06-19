import math
import copy
from collections import OrderedDict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from model.backbones.swin import SwinTransformerBlock
from attention import se_block, cbam_block, eca_block, PAM_CAM_Layer, Att

attention_block = [se_block, cbam_block, eca_block, PAM_CAM_Layer, Att]


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinDecoder(nn.Module):
    def __init__(self, low_level_idx, high_level_idx,
                 input_size, input_dim, num_classes,
                 depth, last_layer_depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, norm_layer, decoder_norm, use_checkpoint):
        self.window_size = window_size
        super().__init__()
        self.low_level_idx = low_level_idx
        self.high_level_idx = high_level_idx

        self.layers_up = nn.ModuleList()
        for i in range(high_level_idx - low_level_idx):
            layer_up = BasicLayer_up(dim=int(input_dim),
                                     input_resolution=(input_size * 2 ** i, input_size * 2 ** i),
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=drop_path_rate,
                                     norm_layer=norm_layer,
                                     upsample=PatchExpand,
                                     use_checkpoint=use_checkpoint)

            self.layers_up.append(layer_up)

        self.last_layers_up = nn.ModuleList()

        for _ in range(low_level_idx + 1):
            i += 1
            last_layer_up = BasicLayer_up(dim=int(input_dim) * 2,
                                          input_resolution=(input_size * 2 ** i, input_size * 2 ** i),
                                          depth=last_layer_depth,
                                          num_heads=num_heads,
                                          window_size=window_size,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=0.0,
                                          norm_layer=norm_layer,
                                          upsample=PatchExpand,
                                          use_checkpoint=use_checkpoint)
            self.last_layers_up.append(last_layer_up)

        i += 1
        self.final_up = PatchExpand(input_resolution=(input_size * 2 ** i, input_size * 2 ** i),
                                    dim=int(input_dim) * 2,
                                    dim_scale=2,
                                    norm_layer=norm_layer)

        if decoder_norm:
            self.norm_up = norm_layer(int(input_dim) * 2)
        else:
            self.norm_up = None
        self.output = nn.Conv2d(int(input_dim) * 2, num_classes, kernel_size=1, bias=False)

        self.Adjust_channel_conv1_1 = nn.Sequential(
            nn.Conv2d(192, 96, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.aspp_GLTB = Block(dim=input_dim, num_heads=8, window_size=self.window_size)

        self.decoder_GLTB1 = Block(dim=input_dim * 2, num_heads=8, window_size=self.window_size)

        self.decoder_GLTB2 = Block(dim=input_dim * 2, num_heads=8, window_size=self.window_size)

        self.FRH = FeatureRefinementHead(96, 96)

    def forward(self, low_level, aspp, fpn_x):
        """
        low_level: B, Hl, Wl, C
        aspp: B, Ha, Wa, C
        """
        B, Hl, Wl, C = low_level.shape

        _, Ha, Wa, _ = aspp.shape

        aspp = aspp.permute(0, 3, 1, 2)
        aspp = self.aspp_GLTB(aspp)
        aspp = aspp.permute(0, 2, 3, 1)

        # aspp = aspp.view(B, Ha * Wa, C)  # torch.Size([1, 256, 96])
        ###########################################################################

        low_level = low_level.view(B, Hl * Wl, C)
        aspp = aspp.view(B, Ha * Wa, C)

        for index, layer in enumerate(self.layers_up):
            if (index == 1):
                break
            aspp = layer(aspp)
        ################进行通道上的堆叠########################
        aspp = aspp.view(aspp.shape[0], int(math.sqrt(aspp.shape[1])), int(math.sqrt(aspp.shape[1])), aspp.shape[2])
        aspp = torch.cat([aspp, fpn_x], dim=3)

        aspp = aspp.view(aspp.shape[0], aspp.shape[3], aspp.shape[1], aspp.shape[2])
        aspp = self.Adjust_channel_conv1_1(aspp)

        aspp = aspp.view(aspp.shape[0], aspp.shape[2], aspp.shape[3], aspp.shape[1])
        aspp = aspp.view(aspp.shape[0], aspp.shape[1] * aspp.shape[2], aspp.shape[3])
        for index, layer in enumerate(self.layers_up):
            if (index == 0):
                continue
            aspp = layer(aspp)
        ################进行通道上的堆叠########################

        ######################################################################################
        #                  _, HaHa, _ = aspp.shape
        #                  Ha = int(math.sqrt(HaHa))
        x = torch.cat([low_level, aspp], dim=-1)

        #                 low_level = low_level.view(B, Hl, Hl, C)  # torch.Size([1, 96, 64, 64])
        #                 aspp = aspp.view(B, Ha, Ha, C)  # torch.Size([1, 96, 64, 64])
        #
        #                 low_level = low_level.permute(0, 3, 1, 2)  # 通道调整到第二维
        #                 aspp = aspp.permute(0, 3, 1, 2)
        #                 x = self.FRH(low_level, aspp)  # torch.Size([1, 192, 64, 64])
        x = x.view(x.shape[0], x.shape[2], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))

        x = self.decoder_GLTB1(x)
        x = x.view(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
        #                 x = x.permute(0, 2, 3, 1)  #通道放到最后
        #                 B, Ha, Ha, C = x.shape
        #                 x = x.view(B, Ha*Ha, C)
        ######################################################################################
        for layer in self.last_layers_up:
            x = layer(x)

        if self.norm_up is not None:
            x = self.norm_up(x)

        B, HaHa, C = x.shape
        Ha = int(math.sqrt(HaHa))
        x = x.view(B, Ha, Ha, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        x = self.decoder_GLTB2(x)
        # x = self.decoder_GLTB2(x)
        x = x.permute(0, 2, 3, 1)  # 通道调整到最后
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)

        x = self.final_up(x)

        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.output(x)

        return x

    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin decoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 1 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    current_k_2 = 'last_layers_up.' + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
                    full_dict.update({current_k_2: v})

            found = 0
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                    else:
                        found += 1

            msg = self.load_state_dict(full_dict, strict=False)
            # print(msg)

            print(f"Decoder Found Weights: {found}")
        else:
            print("none pretrain")

    def load_from_extended(self, pretrained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict = self.state_dict()

        selected_weights = OrderedDict()
        for k, v in model_dict.items():
            # if 'relative_position_index' in k: continue
            if 'blocks' in k:
                name = ".".join(k.split(".")[2:])
                shape = v.shape

                for pre_k, pre_v in pretrained_dict.items():
                    if name in pre_k and shape == pre_v.shape:
                        selected_weights[k] = pre_v

        msg = self.load_state_dict(selected_weights, strict=False)
        found = len(model_dict.keys()) - len(msg.missing_keys)

        print(f"Decoder Found Weights: {found}")


# ------------------------------------------------------------------------------------------------------------------------------------#
import torch.nn.functional as F



class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=96, decode_channels=96):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.Sequential(
            # SeparableConvBN(in_channels, in_channels*2, kernel_size=1),
            # nn.ReLU6()
            ConvBNReLU(in_channels, in_channels * 2, 1)
        )

    def forward(self, x, res):
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x

        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

from timm.models.layers import DropPath, trunc_normal_


class ConvBNReLU1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU1, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels, momentum=0.1),
            nn.ReLU6(inplace=True)

        )


class DepthConvBNReLU1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(DepthConvBNReLU1, self).__init__(
            DepthWiseConv(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                          dilation=dilation, stride=stride),
            norm_layer(out_channels, momentum=0.1),
            nn.ReLU6(inplace=True)
        )


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):

        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    dilation=dilation,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2),
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)

        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        #------------------------------------------------------------------------------------#
        self.branch1 = ConvBNReLU1(in_channels=dim, out_channels=dim, kernel_size=1, stride=1,
                                   dilation=1, bias=True)
        self.branch2 = DepthConvBNReLU1(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                                        dilation=1, bias=True)
        self.branch3 = DepthConvBNReLU1(in_channels=dim, out_channels=dim, kernel_size=4, stride=1,
                                        dilation=2, bias=True)
        self.branch4 = DepthConvBNReLU1(in_channels=dim, out_channels=dim, kernel_size=5, stride=1,
                                        dilation=2, bias=True)
        self.branch_conv1_1 = ConvBNReLU1(dim * 4, dim * 4, kernel_size=1, stride=1, bias=True)
        self.globe_conv = ConvBNReLU1(in_channels=dim, out_channels=dim * 4, kernel_size=1, stride=1,
                                      dilation=1, bias=True)
        self.global_feature = ConvBNReLU1(dim, dim, kernel_size=1, stride=1, bias=True)
        self.branch_adjust = nn.Sequential(
            ConvBNReLU1(dim, dim, kernel_size=1, stride=2, bias=True),
        )
        self.pixelshuffle = nn.PixelShuffle(2)
        #------------------------------------------------------------------------------------#
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # local = self.local2(x) + self.local1(x)
#------------------------------------------------------------------------------------------#
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)

        global_feature = self.global_feature(global_feature)
        global_feature = F.interpolate(global_feature, (x.shape[2], x.shape[3]), None, 'bilinear', True)

        branch = torch.concat([branch1, branch2, branch3, branch4], dim=1) + self.globe_conv(global_feature)
        branch = self.branch_conv1_1(branch)
        branch = self.pixelshuffle(branch)
        branch = self.branch_adjust(branch)
        local = branch

#------------------------------------------------------------------------------------------#
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def build_decoder(input_size, input_dim, config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm

    if config.decoder_name == 'swin':
        return SwinDecoder(
            input_dim=input_dim,
            input_size=input_size,
            low_level_idx=config.low_level_idx,
            high_level_idx=config.high_level_idx,
            num_classes=config.num_classes,
            depth=config.depth,
            last_layer_depth=config.last_layer_depth,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=norm_layer,
            decoder_norm=config.decoder_norm,
            use_checkpoint=config.use_checkpoint
        )


if __name__ == '__main__':
    from config import DecoderConfig

    low_level = torch.randn(2, 96, 96, 96)
    aspp = torch.randn(2, 24, 24, 96)

    decoder = build_decoder(24, 96, DecoderConfig)
    print(sum([p.numel() for p in decoder.parameters()]) / 10 ** 6)

    features = decoder(low_level, aspp)
    print(features.shape)
