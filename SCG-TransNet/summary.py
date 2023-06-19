# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import importlib

import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.SCG_TransNet import SwinDeepLab
if __name__ == "__main__":
    input_shape = [256, 256]
    num_classes = 6
    # backbone        = 'mobilenet'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = importlib.import_module(f'model.configs.swin_224_7_4level')
    model = SwinDeepLab(
        model_config.EncoderConfig,
        model_config.ASPPConfig,
        model_config.DecoderConfig
    ).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)

    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
