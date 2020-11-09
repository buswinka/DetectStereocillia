import src.train
import src.evaluate
from src.dataloader import MaskData, KeypointData
import src.transforms as t
import src.utils
from src.gui import gui


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as Tk
import skimage.io as io

from torch.utils.data import DataLoader
import torchvision

import warnings
import argparse
import sys
import PySimpleGUI as sg

import os.path

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


def main(train_data, epochs, lr,  train_maskrcnn, train_keypoint_rcnn, eval):

    if epochs is not None:
        epochs = int(epochs)

    if train_maskrcnn:
        transforms = torchvision.transforms.Compose([
            t.to_cuda(),
            t.random_h_flip(),
            t.random_v_flip(),
            t.random_affine(),
            t.gaussian_blur(),
            t.random_resize(),
            t.stack_image(),
            t.adjust_brightness(),
            t.adjust_contrast(),
            t.correct_boxes()
        ])
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        print('Loading data... ',end='')
        data = MaskData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=0)
        print('Done')
        print(f'Begining analysis of {len(data)} images: ')

        src.train.train_mask_rcnn(data, epochs=epochs, lr=float(lr))

    if train_keypoint_rcnn:
        transforms = None
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        data = KeypointData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=4)
        print('datalen', len(data))
        src.train.train_keypoint_rcnn(data, epochs=epochs,lr=lr)


    if eval:
        src.evaluate.evaluate(eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='Number of epochs to train.')
    parser.add_argument("--train_mask_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("--train_keypoint_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("-t", "--train_all", help='Do we train a all models from scratch?', action="store_true")
    parser.add_argument("-d", "--train_data", help='Location of training data')
    parser.add_argument("-e", '--eval', help='Path to image file to analyze.')
    parser.add_argument('--lr', help='Learning Rate')
    args = parser.parse_args()
    # If any arg is passed do this thing
    if len(sys.argv) > 1:
        main(args.train_data, args.epochs, args.lr, args.train_mask_rcnn, args.train_keypoint_rcnn, args.eval)
    else:
       gui()

    print(' ')




