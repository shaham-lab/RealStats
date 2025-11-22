import io
import os
import random
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class JPEGCompressionTransform:
    """Apply JPEG compression to an image."""
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)
    

class ImageDataset(Dataset):
    """Custom Dataset for loading images from either a directory or a list of file paths with labels.

    Always returns the original file path for each sample to allow
    per-image processing and caching.
    """

    def __init__(self, image_input, labels=None, transform=None):
        """
        Args:
            image_input (str or list of str): Either a directory path containing images or a list of image file paths.
            labels (list of int or int, optional): List of labels corresponding to the images or a single label for all images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        if isinstance(image_input, str):
            self.image_paths = [os.path.join(image_input, f) for f in os.listdir(image_input) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        elif isinstance(image_input, list):
            self.image_paths = image_input
        else:
            raise TypeError("image_input should be either a directory path (str) or a list of file paths (list).")
        
        if isinstance(labels, list):
            if len(self.image_paths) != len(labels):
                raise ValueError("Length of image paths and labels must match.")
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = [labels] * len(self.image_paths)
        elif labels is None:
            self.labels = [0] * len(self.image_paths)
        else:
            raise TypeError("Labels should be either a list or a single integer value.")
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label, image_path


class ProGanDataset(Dataset):
    """
    Dataset for ProGAN with nested directory structure. Loads all images from '0_real' subdirectories.
    """
    def __init__(self, root_dir, label=0, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class subdirectories with '0_real' and '1_fake'.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path): 
                data_dir = os.path.join(class_path, f'{label}_real') 
                if os.path.isdir(data_dir): 
                    for image_name in os.listdir(data_dir):
                        if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.image_paths.append(os.path.join(data_dir, image_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, image_path


class CocoDataset(Dataset):
    """
    Dataset class for loading COCO samples listed in the CSV file.
    """

    def __init__(self, root_dir, label=0, transform=None, csv_file="list_train.csv"):
        """
        Args:
            root_dir (str): Root directory containing the COCO images (e.g., coco2017/train2017).
            csv_file (str, optional): Path to the CSV file containing image information. Defaults to "train_set/list_train.csv".
            label (int, optional): Label to assign to all samples in this dataset. Defaults to 0 (real samples).
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        self.image_paths = self.data['filename0' if label==0 else 'filename1'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.label, image_path
    

class RealStatsDataset(Dataset):
    """Dataset class for loading samples from CSVs."""

    def __init__(self, root_dir, csv_file, label=0, transform=None, generator=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        csv_path = os.path.join(root_dir, csv_file)
        df = pd.read_csv(csv_path)

        if generator is not None and "fake" in csv_file:
            rel_paths = df["dataset_name"].astype(str) + "/" + df["path"].astype(str)
            df = df[rel_paths.str.startswith(generator)]

        self.data = df

        self.image_paths = [
            os.path.join(self.root_dir, row["dataset_name"], row["path"])
            for _, row in self.data.iterrows()
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label, image_path


class CrossDomainRealDataset(Dataset):
    """Dataset class for mixing reference and shifted real domains.

    The CSV is expected to include a ``path`` column and can optionally include a
    ``dataset_name`` column. When present, ``dataset_name`` is appended between
    the root directory and the relative path to allow mixing samples from
    multiple datasets that share a common parent directory.
    """

    def __init__(
        self,
        root_dir,
        csv_file,
        label=0,
        transform=None,
        dataset_column="dataset_name",
        path_column="path",
    ):
        self.transform = transform
        self.label = label

        csv_path = os.path.join(root_dir, csv_file)
        df = pd.read_csv(csv_path)

        self.image_paths = []
        for _, row in df.iterrows():
            dataset_name = (
                str(row[dataset_column])
                if dataset_column in row and not pd.isna(row[dataset_column])
                else None
            )
            relative_path = str(row[path_column])

            if os.path.isabs(relative_path):
                image_path = relative_path
            elif dataset_name:
                image_path = os.path.join(root_dir, dataset_name, relative_path)
            else:
                image_path = os.path.join(root_dir, relative_path)

            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label, image_path


class EmptyDataset(Dataset):
    """Empty dataset placeholder used when a split is intentionally omitted."""

    def __init__(self, label=1, transform=None):
        self.label = label
        self.transform = transform
        self.image_paths = []

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("EmptyDataset contains no samples")
    

class GlobalPatchDataset(Dataset):
    """Dataset that extracts a specific patch from each image in the original dataset."""
    def __init__(self, original_dataset, patch_size, patch_index):
        self.original_dataset = original_dataset
        self.patch_size = patch_size
        self.patch_index = patch_index

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label, path = self.original_dataset[idx]

        h_patches = image.shape[1] // self.patch_size
        w_patches = image.shape[2] // self.patch_size

        row_idx = self.patch_index // w_patches
        col_idx = self.patch_index % w_patches

        patch = image[
            :,
            row_idx * self.patch_size:(row_idx + 1) * self.patch_size,
            col_idx * self.patch_size:(col_idx + 1) * self.patch_size
        ]

        return patch, label, path
    

class SelfPatchDataset(Dataset):
    """Extract all patches from each image in the dataset."""

    def __init__(self, original_dataset, patch_size):
        """
        Args:
            original_dataset (Dataset): The original dataset providing full images.
            patch_size (int): Size of the square patches.
        """
        self.original_dataset = original_dataset
        self.patch_size = patch_size

    @property
    def image_paths(self):
        return self.original_dataset.image_paths
    
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            patches (torch.Tensor): All patches from the image (shape: N x C x H x W).
            label (int): Label corresponding to the image.
        """
        image, label, path = self.original_dataset[idx]

        h_patches = image.shape[1] // self.patch_size
        w_patches = image.shape[2] // self.patch_size

        patches = []
        for row_idx in range(h_patches):
            for col_idx in range(w_patches):
                patch = image[
                    :, 
                    row_idx * self.patch_size:(row_idx + 1) * self.patch_size,
                    col_idx * self.patch_size:(col_idx + 1) * self.patch_size
                ]
                patches.append(patch)

        patches_tensor = torch.stack(patches)
        return patches_tensor, label, path 


def create_inference_dataset(real_dir, fake_dir, num_samples_per_class, classes='both', shuffle=False):
    """
    Create a balanced dataset for inference by sampling images from real and fake directories.
    Args:
        real_dir (str): Directory containing real images.
        fake_dir (str): Directory containing fake images.
        num_samples_per_class (int): Number of samples per class (-1 to load all samples).
        classes (str): Which classes to include ('both', 'real', or 'fake').

    Returns:
        list: List of tuples (image_path, label), where label is 0 for real and 1 for fake.
    """
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    if num_samples_per_class == -1:
        sampled_real_images = real_images
        sampled_fake_images = fake_images
    else:
        sampled_real_images = random.sample(real_images, min(len(real_images), num_samples_per_class))
        sampled_fake_images = random.sample(fake_images, min(len(fake_images), num_samples_per_class))

    if classes == 'both':
        inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    elif classes == 'fake':
        inference_data = [(img, 1) for img in sampled_fake_images]
    elif classes == 'real':
        inference_data = [(img, 0) for img in sampled_real_images]
    else:
        raise ValueError(f"Invalid classes argument: {classes}. Choose from 'both', 'real', or 'fake'.")

    if shuffle:
        random.shuffle(inference_data)
    return inference_data
