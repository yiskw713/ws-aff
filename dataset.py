import torch

from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import random
import scipy.io

from PIL import Image


class PartAffordanceDataset(Dataset):
    """Part Affordance Dataset"""

    def __init__(self, csv_file, config, transform=None, mode='classify', make_cam_label=False):
        super().__init__()

        self.df = pd.read_csv(csv_file)
        self.config = config
        self.transform = transform
        self.mode = mode    # mode => (training or test)
        self.make_cam_label = make_cam_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        aff_path = self.df.iloc[idx, 1]
        obj_path = self.df.iloc[idx, 2]

        image = Image.open(image_path)
        aff_label = np.load(aff_path)
        obj_label = np.load(obj_path)

        sample = {
            'image': image,
            'obj_label': obj_label,
            'aff_label': aff_label,
        }

        if self.mode == 'train segmentator':
            if self.config.target == 'affordance':
                aff_cam = Image.open(image_path[:-7] + 'aff_cam_label.png')
                sample['aff_cam'] = aff_cam
            elif self.config.target == 'object':
                obj_cam = Image.open(image_path[:-7] + 'obj_cam_label.png')
                sample['obj_cam'] = obj_cam
            else:
                # TODO: error processing
                pass

        if self.mode == 'test':
            label_path = self.df.iloc[idx, 3]
            label = scipy.io.loadmat(label_path)["gt_label"]
            sample['label'] = label

        if self.transform:
            sample = self.transform(sample)

        if self.make_cam_label:
            sample['path'] = image_path

        return sample


''' transforms for pre-processing '''


def crop_center_pil_image(pil_img, crop_height, crop_width):
    w, h = pil_img.size
    return pil_img.crop(((w - crop_width) // 2,
                         (h - crop_height) // 2,
                         (w + crop_width) // 2,
                         (h + crop_height) // 2))


def crop_center_numpy(array, crop_height, crop_width):
    h, w = array.shape
    return array[
        h // 2 - crop_height // 2: h // 2 + crop_height // 2,
        w // 2 - crop_width // 2: w // 2 + crop_width // 2]


def crop_pil_image(pil_img, crop_height, crop_width, top, left):
    return pil_img.crop(
        (left, top, left + crop_width, top + crop_height))


def crop_numpy(array, crop_height, crop_width, top, left):
    return array[top: top + crop_height, left: left + crop_width]


class RandomCrop(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, sample):
        image = sample['image']
        w, h = image.size
        top = np.random.randint(0, h - self.config.crop_height)
        left = np.random.randint(0, w - self.config.crop_width)
        image = crop_pil_image(
            image, self.config.crop_height,
            self.config.crop_width, top, left)
        sample['image'] = image

        if 'aff_cam' in sample:
            aff_cam = sample['aff_cam']
            aff_cam = crop_pil_image(
                aff_cam, self.config.crop_height,
                self.config.crop_width, top, left
            )
            sample['aff_cam'] = aff_cam

        if 'obj_cam' in sample:
            obj_cam = sample['obj_cam']
            obj_cam = crop_pil_image(
                obj_cam, self.config.crop_height,
                self.config.crop_width, top, left
            )
            sample['obj_cam'] = obj_cam

        if 'label' in sample:
            label = sample['label']
            label = crop_numpy(
                label, self.config.crop_height,
                self.config.crop_width, top, left
            )
            sample['label'] = label
        return sample


class CenterCrop(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, sample):
        image = sample['image']
        image = crop_center_pil_image(
            image, self.config.crop_height, self.config.crop_width)
        sample['image'] = image

        if 'aff_cam' in sample:
            aff_cam = sample['aff_cam']
            aff_cam = crop_center_pil_image(
                aff_cam, self.config.crop_height, self.config.crop_width)
            sample['aff_cam'] = aff_cam

        if 'obj_cam' in sample:
            obj_cam = sample['obj_cam']
            obj_cam = crop_center_pil_image(
                obj_cam, self.config.crop_height, self.config.crop_width)
            sample['obj_cam'] = obj_cam

        if 'label' in sample:
            label = sample['label']
            label = crop_center_numpy(
                label, self.config.crop_height, self.config.crop_width)
            sample['label'] = label

        return sample


# when you test the trained model, do not use Resize
class Resize(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, sample):
        image = sample['image']
        image = transforms.functional.resize(
            image, (self.config.height, self.config.width)
        )
        sample['image'] = image
        return sample


class RandomFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            image = transforms.functional.hflip(image)
            sample['image'] = image

            if 'aff_cam' in sample:
                aff_cam = sample['aff_cam']
                aff_cam = transforms.functional.hflip(aff_cam)
                sample['aff_cam'] = aff_cam

            if 'obj_cam' in sample:
                obj_cam = sample['obj_cam']
                obj_cam = transforms.functional.hflip(obj_cam)
                sample['obj_cam'] = obj_cam

            if 'label' in sample:
                label = sample['label']
                label = np.flip(label, axis=0).copy()
                sample['label'] = label
            return sample

        return sample


class RandomRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        super().__init__()
        self.transform = transforms.RandomRotation(
            degrees, resample, expand, center)

    def __call__(self, sample):
        image = sample['image']
        image = self.transform(image)
        sample['image'] = image
        return sample


class ColorChange(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        super().__init__()
        self.transform = transforms.ColorJitter(
            brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image = sample['image']
        image = self.transform(image)
        sample['image'] = image
        return sample


class ToTensor(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        image, obj_label, aff_label = \
            sample['image'], sample['obj_label'], sample['aff_label']

        sample['image'] = transforms.functional.to_tensor(image).float()
        sample['obj_label'] = torch.from_numpy(obj_label).float()
        sample['aff_label'] = torch.from_numpy(aff_label).float()

        if 'aff_cam' in sample:
            aff_cam = sample['aff_cam']
            sample['aff_cam'] = \
                transforms.functional.to_tensor(aff_cam).squeeze().long()

            # in label bg class is not included
            sample['obj_label'] = torch.cat([torch.tensor([0.]), sample['obj_label']])

        if 'obj_cam' in sample:
            obj_cam = sample['obj_cam']
            sample['obj_cam'] = \
                transforms.functional.to_tensor(obj_cam).squeeze().long()

            # in label bg class is not included
            sample['obj_label'] = torch.cat([torch.tensor([0.]), sample['obj_label']])

        if 'label' in sample:
            label = sample['label']
            sample['label'] = torch.from_numpy(label).long()

        return sample


class Normalize(object):
    def __init__(self, mean=[0.2191, 0.2349, 0.3598], std=[0.1243, 0.1171, 0.0748]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = transforms.functional.normalize(image, self.mean, self.std)
        sample['image'] = image
        return sample


def reverse_normalize(x, mean=[0.2191, 0.2349, 0.3598], std=[0.1243, 0.1171, 0.0748]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


'''
obj_list = [
        'bowl',
        'cup',
        'hammer',
        'knife',
        'ladle',
        'mallet',
        'mug',
        'pot',
        'saw',
        'scissors',
        'scoop',
        'shears',
        'shovel',
        'spoon',
        'tenderizer',
        'trowel',
        'turner'
]

aff_list = [
        'grasp',
        'cut',
        'scoop',
        'contain',
        'pound',
        'support',
        'wrap-grasp'
]

'''

'''
# if you want to calculate mean and std of each channel of the images,
# try this code:

data = PartAffordanceDataset('image_class_path.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor()
                                ]))

data_laoder = DataLoader(data, batch_size=10, shuffle=False)

mean = 0
std = 0
n = 0

for sample in data_laoder:
    img = sample['image']
    img = img.view(len(img), 3, -1)
    mean += img.mean(2).sum(0)
    std += img.std(2).sum(0)
    n += len(img)

mean /= n
std /= ns

'''


'''
# if you also want to calculate class weight,
# please try this code

dataset = PartAffordanceDataset('train_with_label.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor()
                                ]))

data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

cnt_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

for sample in data_loader:
    img = sample['label'].numpy()

    num, cnt = np.unique(img, return_counts=True)

    for n, c in zip(num, cnt):
        cnt_dict[n] += c

# cnt_dict
# {0: 1151953630, 1: 14085528, 2: 6604904, 3: 5083312,
#  4: 15579160, 5: 2786632, 6: 3814170, 7: 8105464}

class_num = torch.tensor([1151953630, 14085528, 6604904, 5083312,
                        15579160, 2786632, 3814170, 8105464])
total = class_num.sum().item()
frequency = class_num.float() / total
median = torch.median(frequency)
class_weight = median / frequency

'''
