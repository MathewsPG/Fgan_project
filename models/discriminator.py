import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Road Image Restoration.
    Classifies whether 70x70 overlapping image patches are real clean road scenes
    or synthesized de-rained/de-hazed outputs conditioned on the degraded input image.
    Preserves sharp road surface textures, lane boundaries, and vehicle contours.
    """
    def __init__(self, in_channels=6, features=(64, 128, 256, 512)):
        super(PatchGANDiscriminator, self).__init__()
        # in_channels = 3 (degraded road input) + 3 (clean/restored road target) = 6
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_c, feature, stride=1 if feature == features[-1] else 2)
            )
            in_c = feature

        layers.append(
            nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # Concatenate condition (degraded input) and candidate (real or fake clean)
        xy = torch.cat([x, y], dim=1)
        return self.model(self.initial(xy))


if __name__ == "__main__":
    disc = PatchGANDiscriminator(in_channels=6)
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    preds = disc(x, y)
    print("PatchGAN Discriminator output shape:", preds.shape)
