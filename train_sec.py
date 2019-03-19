import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter

from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize
from dataset import RandomFlip, RandomCrop
from model.deeplabv2 import DeepLabV2
from utils.loss import SeedingLoss, ExpansionLoss, ConstrainToBoundaryLoss


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='choose a device you want to use')

    return parser.parse_args()


''' training '''


def train(model, sample, seed, expand, constrain, optimizer, config, device):
    ''' full supervised learning for segmentation network'''
    model.train()

    x = sample['image']
    x = x.to(device)

    if config.target == 'object':
        y = sample['obj_label']
        cam = sample['obj_cam']
        y = y.to(device)
        cam = cam.to(device)
    elif config.target == 'affordance':
        y = sample['aff_label']
        cam = sample['aff_cam']
        y = y.to(device)
        cam = cam.to(device)
    else:
        # TODO: error processing
        pass

    # h is probability map of each affordance class (N, n_classes, H', W')
    h = model(x)
    seed_loss = seed(h, cam)
    expand_loss = expand(h, y)
    constrain_loss = constrain(x, h, y)

    print(seed_loss)
    print(expand_loss)
    print(constrain_loss)

    loss = seed_loss + expand_loss + constrain_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def one_hot(label, n_classes, dtype, device, requires_grad=True):
    one_hot_label = torch.eye(
        n_classes, dtype=dtype, requires_grad=requires_grad, device=device)[label].transpose(1, 3).transpose(2, 3)
    return one_hot_label


def eval_model(model, test_loader, criterion, config, device):
    ''' calculate the accuracy'''

    model.eval()

    # including background
    intersections = torch.zeros(config.n_classes).to(device)
    unions = torch.zeros(config.n_classes).to(device)
    loss = 0.0

    for sample in test_loader:
        x, y = sample['image'], sample['label']
        _, _, H, W = x.shape

        if config.target == 'object':
            y = torch.where(y > 0, torch.tensor([1]), torch.tensor([0])).long()

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            h = model(x)
            h = F.interpolate(h, (H, W), mode='bilinear')

            loss += criterion(h, y)
            _, ypred = h.max(1)    # y_pred.shape => (N, H, W)

            p = one_hot(
                ypred, config.n_classes, torch.long, device, requires_grad=False)
            t = one_hot(
                y, config.n_classes, torch.long, device, requires_grad=False)

            intersection = torch.sum(p & t, (0, 2, 3))
            union = torch.sum(p | t, (0, 2, 3))

            intersections += intersection.float()
            unions += union.float()

    """ iou[i] is the IoU of class i """
    iou = intersections / unions
    loss /= len(test_loader)

    return iou, loss.item()


''' learning rate scheduler '''


def poly_lr_scheduler(
        optimizer, init_lr, iter, lr_decay_iter=1,
        max_iter=100, power=0.9
):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """

    if iter % lr_decay_iter or iter > max_iter:
        pass
    else:
        lr = init_lr * (1 - iter / max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    """ DataLoader """

    train_data = PartAffordanceDataset(
        CONFIG.train_data,
        config=CONFIG,
        transform=transforms.Compose([
            RandomCrop(CONFIG),
            RandomFlip(),
            ToTensor(CONFIG),
            Normalize()
        ]),
        mode='train segmentator'
    )

    test_data = PartAffordanceDataset(
        CONFIG.test_data,
        config=CONFIG,
        transform=transforms.Compose([
            CenterCrop(CONFIG),
            ToTensor(CONFIG),
            Normalize()
        ]),
        mode='test'
    )

    train_loader = DataLoader(
        train_data, batch_size=CONFIG.batch_size,
        shuffle=True, num_workers=CONFIG.num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_data, batch_size=CONFIG.batch_size,
        shuffle=False, num_workers=CONFIG.num_workers
    )

    print('\n------------Loading Model------------\n')

    model = DeepLabV2(
        n_classes=CONFIG.n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.to(args.device)
    print('Success')

    """ optimizer, criterion """

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    seed = SeedingLoss()
    expand = ExpansionLoss(args.device)
    constrain = ConstrainToBoundaryLoss(args.device)
    criterion = nn.CrossEntropyLoss()

    losses_train = []
    losses_val = []
    class_mean_iou = []
    mean_iou = []
    mean_iou_without_bg = []
    best_mean_iou = 0.0
    print('Start training.')

    for epoch in range(CONFIG.max_epoch):

        poly_lr_scheduler(
            optimizer, CONFIG.learning_rate,
            epoch, max_iter=CONFIG.max_epoch, power=CONFIG.poly_power)

        epoch_loss = 0.0

        for sample in tqdm.tqdm(train_loader, total=len(train_loader)):

            epoch_loss += train(
                model, sample, seed, expand, constrain, optimizer, CONFIG, args.device
            )

        losses_train.append(epoch_loss / len(train_loader))

        # validation
        iou, loss_val = eval_model(
            model, test_loader, criterion, CONFIG, args.device)
        losses_val.append(loss_val)

        class_mean_iou.append(iou)
        mean_iou.append(class_mean_iou[-1].mean().item())
        mean_iou_without_bg.append(class_mean_iou[-1][1:].mean().item())

        if best_mean_iou < mean_iou[-1]:
            best_mean_iou = mean_iou[-1]
            torch.save(
                model.state_dict(), CONFIG.result_path + '/best_accuracy_model.prm')

        if epoch % 50 == 0 and epoch != 0:
            torch.save(
                model.state_dict(), CONFIG.result_path + '/epoch_{}_model.prm'.format(epoch))

        if writer is not None:
            writer.add_scalars("loss", {'loss_train': losses_train[-1],
                                        'loss_val': losses_val[-1]}, epoch)
            writer.add_scalars("loss", {'mean_iou': mean_iou[-1],
                                        'mean_iou_without_bg': mean_iou_without_bg[-1]}, epoch)
            writer.add_scalars("class_IoU", {
                'iou of class 0': class_mean_iou[-1][0],
                'iou of class 1': class_mean_iou[-1][1],
                'iou of class 2': class_mean_iou[-1][2],
                'iou of class 3': class_mean_iou[-1][3],
                'iou of class 4': class_mean_iou[-1][4],
                'iou of class 5': class_mean_iou[-1][5],
                'iou of class 6': class_mean_iou[-1][6],
                'iou of class 7': class_mean_iou[-1][7]}, epoch)

        print(
            'epoch: {}\tloss train: {:.5f}\tloss val: {:.5f}\tmean iou: {:.5f}\tmean iou w/o bg: {:.5f}'
            .format(epoch, losses_train[-1], losses_val[-1], mean_iou[-1], mean_iou_without_bg[-1])
        )

    torch.save(model.state_dict(), CONFIG.result_path + '/final_model.prm')


if __name__ == '__main__':
    main()
