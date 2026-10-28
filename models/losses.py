import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 based perceptual feature loss. Computes multi-layer feature distance
    to preserve high-level road semantics and vehicle contours.
    """
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        except Exception:
            try:
                from torchvision.models import vgg19
                vgg = vgg19(pretrained=True).features
            except Exception as e:
                raise ImportError(f"torchvision VGG19 not accessible: {e}")

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4): # up to relu1_2
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): # up to relu2_2
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18): # up to relu3_4
            self.slice3.add_module(str(x), vgg[x])
        for x in range(18, 27): # up to relu4_4
            self.slice4.add_module(str(x), vgg[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_norm = (pred + 1.0) / 2.0
        target_norm = (target + 1.0) / 2.0

        pred_norm = (pred_norm - self.mean) / self.std
        target_norm = (target_norm - self.mean) / self.std

        h_pred = self.slice1(pred_norm)
        h_target = self.slice1(target_norm)
        loss = 0.1 * F.l1_loss(h_pred, h_target)

        h_pred = self.slice2(h_pred)
        h_target = self.slice2(h_target)
        loss += 0.2 * F.l1_loss(h_pred, h_target)

        h_pred = self.slice3(h_pred)
        h_target = self.slice3(h_target)
        loss += 0.5 * F.l1_loss(h_pred, h_target)

        h_pred = self.slice4(h_pred)
        h_target = self.slice4(h_target)
        loss += 1.0 * F.l1_loss(h_pred, h_target)

        return loss


def gaussian_window(window_size, sigma):
    gauss = torch.tensor([
        -(x - window_size // 2)**2 / float(2 * sigma**2)
        for x in range(window_size)
    ]).exp()
    return gauss / gauss.sum()


def create_window(window_size, channel=3):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1, img2):
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        channel = img1.size(1)
        if channel != self.channel:
            self.window = create_window(self.window_size, channel).to(img1.device)
            self.channel = channel

        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)


class EdgeLoss(nn.Module):
    """
    Sobel-based Edge Loss ensuring critical road boundaries and lane lines remain crisp.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        gx_pred = F.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred_gray, self.sobel_y, padding=1)
        grad_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2 + 1e-6)

        gx_tgt = F.conv2d(target_gray, self.sobel_x, padding=1)
        gy_tgt = F.conv2d(target_gray, self.sobel_y, padding=1)
        grad_tgt = torch.sqrt(gx_tgt ** 2 + gy_tgt ** 2 + 1e-6)

        return F.l1_loss(grad_pred, grad_tgt)
