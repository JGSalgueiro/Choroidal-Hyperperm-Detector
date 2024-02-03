
import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet++ Decoder Block
class UNetPPDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPPDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


# UNet++ Decoder
class UNetPPDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UNetPPDecoder, self).__init__()
        self.num_stages = len(encoder_channels)
        self.decoder_blocks = nn.ModuleList()

        for i in range(self.num_stages - 1, 0, -1):
            in_channels = encoder_channels[i] + decoder_channels[i]
            out_channels = decoder_channels[i]
            decoder_block = UNetPPDecoderBlock(in_channels, out_channels)
            self.decoder_blocks.append(decoder_block)

    def forward(self, x, encoder_features):
        decoder_outputs = []

        for i, decoder_block in enumerate(self.decoder_blocks):
            decoder_output = decoder_block(x)
            decoder_outputs.append(decoder_output)

            if i < len(encoder_features):
                x = torch.cat([decoder_output, encoder_features[i]], dim=1)

        return decoder_outputs[::-1]  # Reverse the order of decoder outputs

