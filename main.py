import src.train
import src.evaluate
from src.dataloader import MaskData
import src.transforms as t
import src.utils
from torch.utils.data import DataLoader
import torchvision

import warnings
import argparse

warnings.filterwarnings("ignore")


def main(train_data, epochs, train_maskrcnn, eval):

    if epochs is not None:
        epochs = int(epochs)

    if train_maskrcnn:
        transforms = torchvision.transforms.Compose([
            t.random_h_flip(),
            t.random_v_flip(),
            t.random_affine(),
            t.adjust_brightness(),
            t.adjust_contrast(),
            t.correct_boxes()
        ])
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        data = MaskData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=4)

        src.train.train_mask_rcnn(data, epochs=epochs)

    if eval:
        src.evaluate.evaluate(eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='Number of epochs to train.')
    parser.add_argument("--train_mask_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("-t", "--train_all", help='Do we train a all models from scratch?', action="store_true")
    parser.add_argument("-d", "--train_data", help='Location of training data')
    parser.add_argument("-e", '--eval', help='Path to image file to analyze.')
    args = parser.parse_args()

    main(args.train_data, args.epochs, args.train_mask_rcnn, args.eval)
