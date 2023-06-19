import copy
from collections import OrderedDict
import torch
from torch import nn

from model.backbones.swin import BasicLayer
from model.cross_attn import CBAMBlock

class SwinASPP(nn.Module):
    def __init__(self, input_size, input_dim, out_dim, cross_attn,
                 depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, 
                 norm_layer, aspp_norm, aspp_activation, start_window_size,
                 aspp_dropout, downsample, use_checkpoint):
        
        super().__init__()
        
        self.out_dim = out_dim
        if input_size == 24:
            self.possible_window_sizes = [4, 6, 8, 12, 24]
        else:
            self.possible_window_sizes = [i for i in range(start_window_size, input_size+1) if input_size%i==0]
            # print(self.possible_window_sizes) [1, 2, 4, 8, 16]
        self.layers = nn.ModuleList()
        for ws in self.possible_window_sizes:
            layer = BasicLayer(dim=int(input_dim),
                               input_resolution=(input_size, input_size),
                               depth=1 if ws==input_size else depth,
                               num_heads=num_heads,
                               window_size=ws,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=downsample,
                               use_checkpoint=use_checkpoint)
            
            self.layers.append(layer)
        
        if cross_attn == 'CBAM':
            self.proj = CBAMBlock(input_dim=len(self.layers)*input_dim*2,
                                  reduction=12, 
                                  input_size=input_size,
                                  out_dim=out_dim)
        else:
            self.proj = nn.Linear(len(self.layers)*input_dim, out_dim)

        # Check if needed
        self.norm = norm_layer(out_dim) if aspp_norm else None
        if aspp_activation == 'relu':
            self.activation = nn.ReLU()
        elif aspp_activation == 'gelu':
            self.activation = nn.GELU()
        elif aspp_activation is None:
            self.activation = None
        
        self.dropout = nn.Dropout(aspp_dropout)

    def forward(self, x, x_convaspp):
        """
        x: input tensor (high level features) with shape (batch_size, input_size, input_size, input_dim)

        returns ...
        """
        B, H, W, C = x.shape
        x = x.view(B, H*W, C)

        features = []

        for layer in self.layers:
            out, _ = layer(x)
            features.append(out)

        features = torch.cat(features, dim=-1)
        features = torch.cat([features, x_convaspp.view(x_convaspp.shape[0],x_convaspp.shape[1]*x_convaspp.shape[2],x_convaspp.shape[3])], dim=-1)
        features = self.proj(features)
        # print(features.shape)
        # Check if needed 
        if self.norm is not None:
            features = self.norm(features)
        if self.activation is not None:
            features = self.activation(features)
        features = self.dropout(features)

        return features.view(B, H, W, self.out_dim)
    
    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.state_dict()
            num_layers = len(self.layers)
            num_pretrained_layers = set([int(k[7]) for k, v in pretrained_dict.items() if 'layers' in k])

            full_dict = copy.deepcopy(pretrained_dict)
            
            layer_dict = OrderedDict()
            
            for i in range(num_layers):
                keys = [item for item in pretrained_dict.keys() if f'layers.{i}' in item]
                for key in keys:
                    for j in num_pretrained_layers:
                        if key in layer_dict: continue
                        # new_k = "layers." + str(i) + k[8:]
                        pre_k = "layers." + str(j) + key[8:]
                        pre_v = pretrained_dict.get(pre_k, None)
                        if pre_v is not None:
                            layer_dict[key] = copy.deepcopy(pre_v)
                    
                        
                        for k in list(layer_dict.keys()):
                            if k in model_dict:
                                if layer_dict[k].shape != model_dict[k].shape:
                                    # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                                    del layer_dict[k]
                            elif k not in model_dict:
                                del layer_dict[k]
            msg = self.load_state_dict(layer_dict, strict=False)
            # print(msg)
            
            print(f"ASPP Found Weights: {len(layer_dict)}")
        else:
            print("none pretrain")


import torch.nn.functional as F
class ConvASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ConvASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation = rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation = 4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation = 8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation = 12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)

        conv3x3_1 = self.branch2(x)

        conv3x3_2 = self.branch3(x)

        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # result = self.conv_cat(feature_cat)
        result = feature_cat
        return result


def build_aspp(input_size, input_dim, out_dim, config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm
    
    if config.aspp_name == 'swin':
        return SwinASPP(
            input_size=input_size,
            input_dim=input_dim,
            out_dim=out_dim,
            depth=config.depth,
            cross_attn=config.cross_attn,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            aspp_norm=config.aspp_norm,
            aspp_activation=config.aspp_activation,
            start_window_size=config.start_window_size,
            aspp_dropout=config.aspp_dropout,
            downsample=config.downsample,
            use_checkpoint=config.use_checkpoint
        )

def build_convaspp(dim_in, dim_out):
    return ConvASPP(
        dim_in,
        dim_out
    )


if __name__ == '__main__':
    from config import ASPPConfig
    
    batch = torch.randn(2, 24, 24, 384)
    model = build_aspp(24, 384, 96, ASPPConfig)

    print("Num of parameters: ", sum([p.numel() for p in model.parameters()])/10**6)
    print(model.possible_window_sizes)

    out = model(batch)
    print(out.shape)


    # for item in out:
    #     print(item.shape)