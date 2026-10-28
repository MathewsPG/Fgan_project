import os
import sys
import shutil
import urllib.request
from PIL import Image, ImageDraw
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_generator import RoadAdverseWeatherSynthesizer

def create_sample_road_dataset(target_dir="dataset", num_samples=120):
    """
    Downloads or prepares paired driving scenes for training and testing.
    Organizes data into standard structure:
    dataset/
      ├── train/
      │   ├── clean/
      │   └── degraded/
      └── test/
          ├── clean/
          └── degraded/
    """
    train_clean = os.path.join(target_dir, "train", "clean")
    train_degraded = os.path.join(target_dir, "train", "degraded")
    test_clean = os.path.join(target_dir, "test", "clean")
    test_degraded = os.path.join(target_dir, "test", "degraded")

    os.makedirs(train_clean, exist_ok=True)
    os.makedirs(train_degraded, exist_ok=True)
    os.makedirs(test_clean, exist_ok=True)
    os.makedirs(test_degraded, exist_ok=True)

    synthesizer = RoadAdverseWeatherSynthesizer()

    # Curated set of high-resolution royalty-free autonomous driving & road dashcam frames
    road_image_urls = [
        "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1508974239320-0a029497e820?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=800&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1486006920555-c77dce18193b?w=800&auto=format&fit=crop&q=80",
    ]

    print(f"[*] Initializing road dataset generator at '{target_dir}'...")
    raw_clean_dir = os.path.join(target_dir, "raw_clean_temp")
    os.makedirs(raw_clean_dir, exist_ok=True)

    downloaded_images = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    for idx, url in enumerate(road_image_urls):
        try:
            req = urllib.request.Request(url, headers=headers)
            raw_path = os.path.join(raw_clean_dir, f"base_{idx:02d}.jpg")
            with urllib.request.urlopen(req, timeout=10) as response, open(raw_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            img = Image.open(raw_path).convert("RGB")
            img = img.resize((512, 384), Image.Resampling.LANCZOS)
            downloaded_images.append(img)
            print(f" -> Fetched road template {idx+1}/{len(road_image_urls)}")
        except Exception as e:
            print(f" -> Could not fetch image {idx}: {e}")

    # Fallback procedural generation if offline
    if len(downloaded_images) == 0:
        print("[!] Generating procedural autonomous driving road templates offline...")
        for i in range(5):
            h, w = 384, 512
            road = np.zeros((h, w, 3), dtype=np.uint8)
            road[:int(h*0.5), :] = [180 + i*10, 160, 140]
            road[int(h*0.5):, :] = [50 + i*5, 50 + i*5, 55 + i*5]
            img = Image.fromarray(road)
            draw = ImageDraw.Draw(img)
            draw.line([(w//2 - 20, int(h*0.5)), (w//4, h)], fill=(255, 255, 255), width=4)
            draw.line([(w//2 + 20, int(h*0.5)), (3*w//4, h)], fill=(255, 255, 0), width=4)
            downloaded_images.append(img)

    print(f"[*] Generating {num_samples} paired road scenes (rain streaks + atmospheric haze)...")
    train_count = int(num_samples * 0.8)

    for i in range(num_samples):
        base_img = downloaded_images[i % len(downloaded_images)]
        img_np = np.array(base_img)

        if np.random.rand() > 0.5:
            img_np = np.fliplr(img_np)
        
        h, w, _ = img_np.shape
        top = np.random.randint(0, max(1, h - 256))
        left = np.random.randint(0, max(1, w - 256))
        crop_clean = img_np[top:top+256, left:left+256]

        beta = np.random.uniform(0.6, 1.8)
        length = np.random.randint(15, 35)
        angle = np.random.uniform(70, 110)
        density = np.random.uniform(0.008, 0.03)

        hazy = synthesizer.add_haze(crop_clean, beta=beta)
        compound = synthesizer.add_rain_streaks(hazy, length=length, angle=angle, density=density)

        is_train = i < train_count
        split_dir = "train" if is_train else "test"
        sample_name = f"road_pair_{i:04d}.png"

        Image.fromarray(crop_clean).save(os.path.join(target_dir, split_dir, "clean", sample_name))
        Image.fromarray(compound).save(os.path.join(target_dir, split_dir, "degraded", sample_name))

    shutil.rmtree(raw_clean_dir, ignore_errors=True)

    print(f"[+] Paired Road Dataset successfully generated!")
    print(f"    Train pairs: {train_count} (dataset/train/)")
    print(f"    Test pairs:  {num_samples - train_count} (dataset/test/)")


if __name__ == "__main__":
    create_sample_road_dataset()
