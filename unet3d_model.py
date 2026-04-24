import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channel),
            nn.Softplus()
        )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channel),
            nn.Softplus()
        )

    def forward(self, feature_map):
        mid = self.input(feature_map)
        res = self.output(mid)
        return res


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()

        self._input = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.Softplus()
        )

        self._mid = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channel),
            nn.Softplus()
        )

        self._output = nn.Sequential(
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channel),
            nn.Softplus()
        )

    def forward(self, feature_map, skip):
        x = self._input(feature_map)
        mid = self._mid(torch.cat([x, skip], dim=1))
        res = self._output(mid)
        return res


class Unet(nn.Module):
    def __init__(self, depth, base, init_input=1, init_output=1):
        super(Unet, self).__init__()
        self.depth = depth
        self.input = Encoder(init_input, base)

        # Adjusting the scaling factor to 2 to prevent channels from exploding
        self.encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                                     Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                       for i in range(depth)])

        self.decoders = nn.ModuleList([Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                       for i in range(depth, 0, -1)])

        self.output = nn.Conv3d(base, init_output, 1, 1, 0)

    def forward(self, x):
        skips = []
        inEncoder = self.input(x)
        skips.append(inEncoder)

        for encoder in self.encoders:
            inEncoder = encoder(inEncoder)
            skips.append(inEncoder)

        inDecoder = inEncoder
        skips.pop()

        for decoder in self.decoders:
            inDecoder = decoder(inDecoder, skips.pop())

        return self.output(inDecoder)


# Example usage
if __name__ == "__main__":
    depth = 4
    base = 32
    INPUT_CHANNEL = INIT_CHANNEL = 1
    OUTPUT_CHANNEL = 1

    model = Unet(depth=depth, base=base, init_input=1, init_output=1)

    print(model)
