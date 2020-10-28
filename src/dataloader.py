import torch
import PIL
import numpy as np
from torchvision import datasets
from torch.utils.data.dataset import Dataset
import glob
import pandas
import skimage.io
import skimage.draw
import json
import torchvision.transforms.functional as TF

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

class MaskData(Dataset):
    def __init__(self, basedir, transforms):
        files = glob.glob(basedir + '*.csv')

        self.transforms = transforms
        self.masks = []
        self.images = []
        self.labels = []
        self.boxes = []

        for f in files:
            df = pandas.read_csv(f)
            image = TF.to_tensor(PIL.Image.open(f[:-4:] +  '.png'))
            image = torch.cat((image,image,image), dim=0)
            mask = torch.zeros((len(df), image.shape[1], image.shape[2]), dtype=torch.bool)
            labels = []
            for l in range(len(df)):
                d = json.loads(df['region_shape_attributes'][l])
                label = json.loads(df['region_attributes'][l])
                try:
                    labels.append(int(label['stereocillia']))
                except KeyError:
                    labels.append(1)

                x = d['all_points_x']
                y = d['all_points_y']
                box = torch.tensor([np.min(x), np.min(y), np.max(x), np.max(y)])
                xx, yy = skimage.draw.polygon(x,y)
                mask[l, yy, xx] = 1
                if l == 0:
                    boxes = box.unsqueeze(0)
                else:
                    boxes = torch.cat((boxes, box.unsqueeze(0)), dim=0)

            for l in range(mask.shape[0]):
                if mask[l,:,:].max() == 0:
                    raise ValueError('Mask is jank')

            self.labels.append(torch.tensor(labels))
            self.masks.append(mask)
            self.boxes.append(boxes)
            self.images.append(image)

    def __getitem__(self, item):
        data_dict ={'boxes': self.boxes[item],
                    'labels': self.labels[item],
                    'masks': self.masks[item],
                    'image': self.images[item]}

        data_dict = self.transforms(data_dict)

        return data_dict['image'], data_dict

    def __len__(self):
        return len(self.images)

class BundleData(Dataset):
    def __init__(self, basedir):
        files = glob.glob(basedir + '*.csv')

        self.images = []
        self.labels = []
        self.boxes = []

        labels = []
        for f in files:
            df = pandas.read_csv(f)
            image = skimage.io.imread(f[:-4:] +  '.png')
            labels = []
            boxes = []

            for l in range(len(df)):
                d = json.loads(df['region_shape_attributes'][l])
                label = json.loads(df['region_attributes'][l])
                labels.append(1)

                x = d['x']
                y = d['y']
                width = d['width']
                height = d['height']

                box = [x,y,x+width, y+height]
                boxes.append(box)

            self.labels.append(labels)
            self.boxes.append(boxes)
            self.images.append(image)

    def __getitem__(self, item):
        data_dict ={'boxes': torch.tensor(self.boxes[item]),
                    'labels': torch.tensor(self.labels[item])}

        im = self.images[item]

        image = torch.zeros(1, 3, im.shape[0], im.shape[1])

        image[0,0,:,:]=torch.from_numpy(im)
        image[0,1,:,:]=torch.from_numpy(im)
        image[0,2,:,:]=torch.from_numpy(im)

        return image, data_dict

    def __len__(self):
        return len(self.images)
