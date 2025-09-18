import requests
import os
import json
from tqdm import tqdm
import numpy as np


class QuickDrawDownloader:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Quick Draw categories - you can modify this list
        self.categories = [
            'apple', 'banana', 'book', 'car', 'cat', 'dog', 'house',
            'tree', 'sun', 'moon', 'star', 'cloud', 'flower', 'heart',
            'airplane', 'bicycle', 'bird', 'boat', 'bridge', 'bus',
            'clock', 'computer', 'eye', 'face', 'guitar', 'hammer',
            'hat', 'key', 'laptop', 'lightning', 'line', 'mountain',
            'pencil', 'phone', 'pizza', 'rain', 'river', 'snowflake',
            'spoon', 'sword', 't-shirt', 'train', 'umbrella', 'wheel'
        ]

    def download_category(self, category, max_samples=10000):
        """Download drawings for a specific category"""
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            file_path = os.path.join(self.data_dir, f"{category}.npy")

            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with open(file_path, 'wb') as f, tqdm(
                    desc=f"Downloading {category}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)

            # Limit samples if needed
            if max_samples:
                data = np.load(file_path)
                if len(data) > max_samples:
                    data = data[:max_samples]
                np.save(file_path, data)

            print(f"Downloaded {category}: {len(data)} samples")
            return True

        except Exception as e:
            print(f"Error downloading {category}: {e}")
            return False

    def download_all_categories(self, max_samples_per_category=10000):
        """Download all categories"""
        success_count = 0

        for category in self.categories:
            if self.download_category(category, max_samples_per_category):
                success_count += 1

        print(f"Successfully downloaded {success_count}/{len(self.categories)} categories")

        # Save categories list
        with open('data/categories.txt', 'w') as f:
            for category in self.categories:
                f.write(f"{category}\n")

    def get_downloaded_categories(self):
        """Get list of successfully downloaded categories"""
        downloaded = []
        for category in self.categories:
            file_path = os.path.join(self.data_dir, f"{category}.npy")
            if os.path.exists(file_path):
                downloaded.append(category)
        return downloaded


if __name__ == "__main__":
    downloader = QuickDrawDownloader()
    downloader.download_all_categories(max_samples_per_category=10000)