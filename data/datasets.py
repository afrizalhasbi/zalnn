import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from .base import BaseDataset, ImageDataset, PILImageDataset
import cv2
import asyncio

class InterpolatedImageDataset(ImageDataset):
    def __init__(self, data, labels=None):
        super(InterpolatedImageDataset, self).__init__(data, labels)
        assert not self.labels, "Interpolated image dataset does not accept labels"

    def __preprocess__(self, datum, label=None):
        original = self.__load_image__(datum)
        degraded_image = self.interpolate(original)

        degraded_image = self.to_torch(degraded_image)
        original = self.to_torch(original)
        return original, degraded_image, torch.tensor(False)

    def to_torch(self, image):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image

class PILInterpolatedImageDataset(PILImageDataset):
    def __init__(self, data, labels=None):
        super(PILInterpolatedImageDataset, self).__init__(data, labels)
        assert not self.labels, "Interpolated image dataset does not accept labels"

    def __preprocess__(self, datum, label=None):
        to_tensor = transforms.ToTensor()

        original = self.__load_image__(datum)
        original = self.prepare(original)
        degraded = self.interpolate(original)
        original, degraded = to_tensor(original), to_tensor(degraded)

        return original, degraded, torch.tensor(False)

    def prepare(self, image):
        prepare = transforms.Compose([
            transforms.Lambda(lambda x: self.evenize(x)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            ])
        return prepare(image)

    # def prepare_for_original(self, image):
        # prepare = transforms.Compose([
            # transforms.Lambda(lambda x: self.evenize(x)),
            # transforms.Lambda(lambda x: self.pad_image(x)),
            # transforms.Lambda(lambda x: x.convert('RGB')),
            # transforms.ToTensor()
        # ])
        # return prepare(image)

    # def prepare_for_degradation(self, image):
        # prepare = transforms.Compose([
            # transforms.Lambda(lambda x: self.evenize(x)),
            # transforms.Lambda(lambda x: self.pad_image(x)),
            # transforms.Lambda(lambda x: self.interpolate(x)),
            # transforms.Lambda(lambda x: x.convert('RGB')),
            # transforms.ToTensor()
        # ])
        # return prepare(image)

class ClassificationImage(ImageDataset):
    def __init__(self, data, labels):
        super(ImageDataset, self).__init__(data, labels)
        assert self.labels, "Classification image dataset must have labels"
        self.num_labels = len(set(labels))

    def __preprocess__(self, datum, label):
        datum = self.__load_image__(datum)
        label = self.__load_image__(label)
        return datum, label

class Seq2SeqText(BaseDataset):
    pass

class AutoregressiveText(BaseDataset):
    pass

class Text2Image(BaseDataset):
    pass
