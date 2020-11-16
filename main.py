import src.train
import src.evaluate
from src.dataloader import MaskData, KeypointData
import src.transforms as t
import src.utils
from src.gui import gui

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib
import warnings
import argparse
import sys

from typing import Optional

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


def main(train_maskrcnn: bool = False, train_keypoint_rcnn: bool = False, eval_model: Optional[str] = False,
         train_data: Optional[str] = None, epochs: Optional[int] = None, lr: Optional[float] = None,
         pretrained: Optional[str] = None):

    """
    Handles delegation of tasks to various other scripts based on command line arguments from argparse

    :param train_data: str - path to training data
    :param epochs: int - number of epochs to train for
    :param lr: float - learning rate of model
    :param train_maskrcnn: bool - if true runs training script for maskrcnn
    :param train_keypoint_rcnn: bool - if true runs training script for keypoint_rcnn
    :param eval_model: str - path to image to be evaluated. Returns nothing.
    :param pretrained: str - path to pretrained maskrcnn model for transfer learning while training
    :return:
    """

    if epochs is not None:
        epochs = int(epochs)

    if train_maskrcnn:
        transforms = torchvision.transforms.Compose([
            t.to_cuda(),
            t.random_h_flip(),
            t.random_v_flip(),
            t.random_affine(),
            t.gaussian_blur(kernel_targets=torch.tensor([3, 5, 7, 9, 11, 13])),
            t.random_resize(scale=(250, 1440)),
            t.stack_image(),
            t.adjust_brightness(),
            t.adjust_contrast(),
            t.correct_boxes()
        ])
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        print('Loading data... ', end='')
        data = MaskData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=0)
        print('Done')
        print(f'Begining analysis of {len(data)} images: ')

        src.train.train_mask_rcnn(data, epochs=epochs, lr=float(lr), pretrained=pretrained)

    if train_keypoint_rcnn:
        transforms = None
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        data = KeypointData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=4)
        print('datalen', len(data))
        src.train.train_keypoint_rcnn(data, epochs=epochs, lr=lr)


    if eval_model:
        src.evaluate.evaluate()(eval_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='Number of epochs to train.')
    parser.add_argument("--train_mask_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("--pretrained_model", help='Path location of pretrained model')
    parser.add_argument("--train_keypoint_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("-t", "--train_all", help='Do we train a all models from scratch?', action="store_true")
    parser.add_argument("-d", "--train_data", help='Location of training data')
    parser.add_argument("-e", '--eval_model', help='Path to mask file to analyze.')
    parser.add_argument('--lr', help='Learning Rate')
    args = parser.parse_args()

    # If any arg is passed do this thing
    if len(sys.argv) > 1:
        main(args.train_data, args.epochs, args.lr, args.train_mask_rcnn, args.train_keypoint_rcnn, args.eval, args.pretrained_model)
    else:
        gui()

    print(' ')




