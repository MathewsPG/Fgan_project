import os
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class RoadAdverseWeatherSynthesizer:
    """
    Physics-based generator for realistic Rain Streaks and Atmospheric Haze/Fog
    specifically tailored for autonomous driving road viewpoints.
    
    Implements:
    1. Atmospheric Scattering Model: I(x) = J(x)t(x) + A(1 - t(x)) with perspective depth estimation.
    2. Motion-blurred Directional Rain Streak field with variable intensity and mist.
    3. Compound Adverse Weather: simultaneous rain streaks + fog/haze.
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def estimate_road_depth_map(self, h, w):
        """
        Synthesizes a realistic autonomous driving depth map:
        - Bottom of the image (ego vehicle hood / nearby road surface): low depth
        - Center horizon line (far vehicles, road vanishing point): high depth
        - Upper region (sky / trees / overpasses): high depth
        """
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        x_coords = np.linspace(0, 1, w)[np.newaxis, :]
        
        horizon = 0.50
        road_mask = (y_coords >= horizon)
        road_depth = 1.0 - (y_coords - horizon) / (1.0 - horizon)
        road_depth = np.clip(road_depth, 0.05, 1.0)
        
        sky_depth = np.ones((h, w), dtype=np.float32)
        depth = np.where(road_mask, road_depth, sky_depth)
        lateral_var = 0.15 * np.sin(np.pi * x_coords)
        depth = np.clip(depth + lateral_var, 0.05, 1.0)
        return depth

    def add_haze(self, img_np, beta=None, atmospheric_light=None):
        """
        Applies Koschmieder's Atmospheric Scattering model.
        img_np: uint8 RGB image in range [0, 255]
        beta: scattering coefficient (0.6 - 1.8)
        atmospheric_light: scalar or 3-tuple [0.75 - 0.95]
        """
        h, w, c = img_np.shape
        img = img_np.astype(np.float32) / 255.0
        
        if beta is None:
            beta = np.random.uniform(0.7, 1.6)
        if atmospheric_light is None:
            A = np.random.uniform(0.75, 0.95)
        else:
            A = atmospheric_light

        depth = self.estimate_road_depth_map(h, w)
        transmission = np.exp(-beta * depth)[:, :, np.newaxis]
        transmission = np.clip(transmission, 0.08, 1.0)

        hazy = img * transmission + A * (1.0 - transmission)
        hazy = np.clip(hazy * 255.0, 0, 255).astype(np.uint8)
        return hazy

    def add_rain_streaks(self, img_np, length=None, angle=None, density=None):
        """
        Synthesizes directional motion-blurred rain streaks.
        """
        h, w, c = img_np.shape
        if length is None:
            length = np.random.randint(15, 35)
        if angle is None:
            angle = np.random.uniform(75, 105)
        if density is None:
            density = np.random.uniform(0.008, 0.025)

        num_drops = int(h * w * density)
        rain_mask = np.zeros((h, w), dtype=np.float32)
        ypoints = np.random.randint(0, h, num_drops)
        xpoints = np.random.randint(0, w, num_drops)
        rain_mask[ypoints, xpoints] = np.random.uniform(0.6, 1.0, num_drops)

        ksize = length if length % 2 == 1 else length + 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        center = ksize // 2

        for step in range(-center, center + 1):
            kx = int(center + step * dx)
            ky = int(center + step * dy)
            if 0 <= kx < ksize and 0 <= ky < ksize:
                kernel[ky, kx] = 1.0

        if kernel.sum() > 0:
            kernel /= kernel.sum()

        if HAS_CV2:
            streaks = cv2.filter2D(rain_mask, -1, kernel)
            streaks = cv2.GaussianBlur(streaks, (3, 3), 0)
        else:
            # Fallback pure NumPy streak drawing
            streaks = np.zeros_like(rain_mask)
            for y, x in zip(ypoints, xpoints):
                val = rain_mask[y, x]
                for step in range(-center, center + 1):
                    ny = int(y + step * dy)
                    nx = int(x + step * dx)
                    if 0 <= ny < h and 0 <= nx < w:
                        streaks[ny, nx] = max(streaks[ny, nx], val * 0.8)

        streaks = np.clip(streaks * 1.8, 0, 1.0)[:, :, np.newaxis]
        rain_color = np.array([210, 220, 235], dtype=np.float32)
        img_float = img_np.astype(np.float32)
        rainy = img_float * (1.0 - 0.65 * streaks) + rain_color * (0.65 * streaks)
        rainy = np.clip(rainy, 0, 255).astype(np.uint8)
        return rainy

    def synthesize_compound_weather(self, img_np):
        """
        Simultaneously injects atmospheric haze and road rain streaks.
        """
        hazy = self.add_haze(img_np)
        compound = self.add_rain_streaks(hazy)
        return compound
