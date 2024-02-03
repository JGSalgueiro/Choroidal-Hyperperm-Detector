from .UNET import UNet
from .unetpp import UNetPlusPlus
from .resunet import ResUNet
from .seuneter import SeUNet
from .drunet import DRUNet
from .unetpp_decoder import UNetPPDecoder
from .seunter_2 import SeUNet_multi

def get_model(model_name):
    if model_name == 'unet':
        return UNet()
    elif model_name == 'unet++':
        return UNetPlusPlus(in_channels=1, out_channels=1)
    elif model_name == 'resunet':
        return ResUNet(in_channels=1, out_channels=1)
    elif model_name == 'seuneter':
        return SeUNet(in_channels=1, out_channels=1)
    elif model_name == 'drunet':
        return DRUNet(in_channels=1, out_channels=1)
    elif get_model == 'unet++_decoder':
        return UNetPPDecoder(in_channels=1, out_channels=1)
    elif model_name == 'seuneter_multi':
        return SeUNet_multi(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

