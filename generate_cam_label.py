import torch

from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import glob
import numpy as np
import yaml
import tqdm

from addict import Dict
from PIL import Image

from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize
from dataset import Resize, RandomFlip, RandomRotate, RandomCrop, reverse_normalize
from model.drn import drn_c_58
from model.drn_max import drn_c_58_max, drn_d_105_max
from utils.cam import CAM, GradCAM


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='make initial label for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='choose a device you want to use')

    return parser.parse_args()


# use color palette of an image in pascal voc
voc_img = Image.open("sample.png")
palette = voc_img.getpalette()


def save_cam(wrapped_model, sample, device):
    img = sample['image'].to(device)

    # calculate cams
    obj_label, aff_label = wrapped_model.get_label(
        img, sample['obj_label'], sample['aff_label'])

    image = Image.fromarray(obj_label.astype(np.uint8))
    image.putpalette(palette)
    image.save(sample['path'][0][:-7] + 'obj_cam_label.png')

    image = Image.fromarray(aff_label.astype(np.uint8))
    image.putpalette(palette)
    image.save(sample['path'][0][:-7] + 'aff_cam_label.png')


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(
        yaml.safe_load(open(args.config)))

    """ DataLoader """
    train_transform = transforms.Compose([
        ToTensor(CONFIG),
        Normalize()
    ])

    train_data = PartAffordanceDataset(
        CONFIG.train_data, config=CONFIG, transform=train_transform, mode='test', make_cam_label=True)

    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=True, num_workers=2)

    """ Load Model """
    if CONFIG.model == 'drn_c_58':
        print(CONFIG.model + "will be used")
        model = drn_c_58(
            pretrained=False, num_obj=CONFIG.obj_classes, num_aff=CONFIG.aff_classes)
    elif CONFIG.model == 'drn_c_58_max':
        print(CONFIG.model + "will be used")
        model = drn_c_58_max(
            pretrained=False, num_obj=CONFIG.obj_classes, num_aff=CONFIG.aff_classes)
    elif CONFIG.model == 'drn_d_105_max':
        print(CONFIG.model + "will be used")
        model = drn_d_105_max(
            pretrained=False, num_obj=CONFIG.obj_classes, num_aff=CONFIG.aff_classes)
    else:
        print(
            'Cannot match exitsting models with the model in config. drn_c_58 will be used.')
        model = drn_c_58(
            pretrained=True, num_obj=CONFIG.obj_classes, num_aff=CONFIG.aff_classes)
    print('Success\n')

    state_dict = torch.load(CONFIG.result_path + '/best_accuracy_model.prm',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    target_layer_obj = model.obj_conv
    target_layer_aff = model.aff_conv

    # choose CAM or GradCAM
    wrapped_model = CAM(model, target_layer_obj, target_layer_aff)
    # wrapped_model = GradCAM(model, target_layer_obj, target_layer_aff)

    for sample in tqdm.tqdm(train_loader, total=len(train_loader)):
        save_cam(wrapped_model, sample, args.device)


if __name__ == '__main__':
    main()
