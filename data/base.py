import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import pandas as pd
from PIL import Image
import torch

class BaseDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels if labels else None
        if self.labels:
            assert len(self.data) == len(self.labels), "Data and labels length must be the same"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels:
            datum, label =  self.data[idx], self.labels[idx]
        else:
            datum, label =  self.data[idx], torch.tensor(False)
        inputs, preprocessed_inputs, targets = self.__preprocess__(datum, label)
        return inputs, preprocessed_inputs, targets

    def __preprocess__(self, datum, label):
        raise NotImplementedError

    @classmethod
    def from_parquet(cls, dspath, datacol, labelcol=None):
        dataset = pd.read_parquet(dspath)
        dataset = dataset.sample(1, random_state=111)
        data = dataset[datacol]
        if labelcol:
            labels = dataset[labelcol]
        else:
            labels = None
        return cls(data, labels)

    @classmethod
    def from_csv(cls, dspath, datacol, labelcol=None):
        dataset = pd.read_csv(dspath)
        dataset = dataset.sample(1, random_state=111)
        data = dataset[datacol]
        if labelcol:
            labels = dataset[labelcol]
        else:
            labels = None
        return cls(data, labels)

    @classmethod
    def from_hf(cls, dspath, datacol, labelcol=None):
        try:
            dataset = load_from_disk(dspath)
        except:
            dataset = load_dataset(dspath)
        try:
            dataset = dataset['train']
        except:
            pass
        dataset = dataset.shuffle(111)
        data = dataset[datacol]
        if labelcol:
            labels = dataset[labelcol]
        else:
            labels = None
        return cls(data, labels)

class ImageDataset(BaseDataset):
    def __init__(self, data, labels=None):
        super(ImageDataset, self).__init__(data, labels)

    def evenize(self, image):
        h, w = image.shape[:2]  # cv2 uses (h,w) order
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        if new_w != w or new_h != h:
            image = cv2.resize(image, (new_w, new_h))
        return image

    def interpolate(self, image):
        scale_factor = random.randint(2, 3)
        resample_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
        resample = random.choice(resample_methods)
        h, w = image.shape[:2]
        resized = cv2.resize(image, (w // scale_factor, h // scale_factor), interpolation=resample)
        resized = cv2.resize(resized, (w, h), interpolation=resample)
        
        return resized
    
    def random_crop(self, image):
        crop_size=(512, 512)
        h, w = image.shape[:2]
        if h < crop_size[0] or w < crop_size[1]:
            return self.resize_image(image)
        x = np.random.randint(0, w - crop_size[1])
        y = np.random.randint(0, h - crop_size[0])
        return image[y:y+crop_size[0], x:x+crop_size[1]]

    def resize_image(self, image, target_size = (512, 512)):
        h, w = image.shape[:2]
        ratio = min(target_size[0]/w, target_size[1]/h)
        new_size = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

        return image

    def __load_image__(self, path):
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.random_crop(image)
        except:
            return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

class PILImageDataset(BaseDataset):
    def __init__(self, data, labels=None):
        super(PILImageDataset, self).__init__(data, labels)

    def evenize(self, image):
        w, h = image.size
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h))
        return image

    def interpolate(self, image):
        scale_factor = random.randint(2, 3)
        resample_methods = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]
        resample = random.choice(resample_methods)  # Random resample method
        ori_width, ori_height = image.width, image.height
        image = image.resize(
            (ori_width // scale_factor, image.height // scale_factor), resample=resample
        )
        image = image.resize((ori_width, ori_height), resample=resample)
        return image
    
    def resize_image(self, image, target_size = (1080, 1080)):
        ratio = min(target_size[0]/image.width, target_size[1]/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def pad_image(self, image, target_size = (1080, 1080)):
        w, h = image.size
        if target_size[0] < h or target_size[1] < w:
            return image
            
        image = np.array(image)
        pad_h, pad_w = target_size[0] - h, target_size[1] - w
        padded = np.pad(image, [(pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2), (0,0)], mode='constant')
        return Image.fromarray(padded)
    
    def __load_image__(self, path):
        image = Image.open(path)
        return self.resize_image(image)

class TextDataset(BaseDataset):
    def __init__(self, data, labels=None):
        super(TextDataset, self).__init__(data, labels)
