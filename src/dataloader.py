import torch
import matplotlib.pyplot as plt
import os.path
import PIL
import numpy as np
from torchvision import datasets
from torch.utils.data.dataset import Dataset
import src.transforms as t
import glob
import pandas
import skimage.io
import skimage.draw
import json
import torchvision.transforms.functional as TF

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

class MaskData(Dataset):
    def __init__(self, basedir, transforms):

        files = glob.glob(os.path.join(basedir,'data.csv'))

        self.transforms = transforms
        self.masks = []
        self.images = []
        self.labels = []
        self.boxes = []

        data_frame = pandas.read_csv(files[0])
        image_names = data_frame['filename'].unique()

        for im_name in image_names:
            df = data_frame[data_frame['filename'] == im_name]

            if len(df) <= 1:
                continue

            im_path = os.path.join(basedir, im_name)
            try:
                image = TF.to_tensor(PIL.Image.open(im_path))
            except:
                continue

            # mask = torch.cat((mask, mask, mask), dim=0)
            mask = torch.zeros((len(df), image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = []


            for l in range(len(df)):
                d = json.loads(df['region_shape_attributes'].to_list()[l])
                label = json.loads(df['region_attributes'].to_list()[l])
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
                    'mask': self.images[item]}

        data_dict = self.transforms(data_dict)

        return data_dict['mask'], data_dict

    def __len__(self):
        return len(self.images)


class BundleData(Dataset):
    def __init__(self, basedir):

        files = glob.glob(os.path.join(basedir,'*.csv'))
        self.images = []
        self.labels = []
        self.boxes = []

        labels = []
        for f in files:
            df = pandas.read_csv(f)
            image = TF.to_tensor(PIL.Image.open(f[:-4:] +  '.png'))
            image = torch.cat((image, image, image), dim=0)
            labels = []
            boxes = []

            for l in range(len(df)):
                d = json.loads(df['region_shape_attributes'][l])
                labels.append(1)

                x = d['x']
                y = d['y']
                width = d['width']
                height = d['height']

                box = torch.tensor([[x,y,x+width, y+height]])
                if l == 0:
                    boxes = box
                else:
                    boxes = torch.cat((boxes, box), dim=0)

            self.labels.append(labels)
            self.boxes.append(boxes)
            self.images.append(image)

    def __getitem__(self, item):
        data_dict ={'boxes': torch.tensor(self.boxes[item]),
                    'labels': torch.tensor(self.labels[item])}

        im = self.images[item]

        return im, data_dict

    def __len__(self):
        return len(self.images)


class ChunjieData(Dataset):
    def __init__(self, basedir):
        files = glob.glob(basedir + 'Markers_Counter Window - *.jpg')

        self.images = []
        self.labels = []
        self.boxes = []

        self.shape = torch.tensor([[0,1,1]])

        labels = []
        for f in files:
            image_point = TF.to_tensor(PIL.Image.open(f))
            filename = os.path.splitext(f)[0]
            filename = filename.replace('Markers_Counter Window - ','')

            try:
                image_data = TF.to_tensor(PIL.Image.open(filename+'.tif'))
            except FileNotFoundError:
                image_data = TF.to_tensor(PIL.Image.open(filename+'.jpg'))


            image_data = torch.cat((image_data, image_data, image_data), dim=0)
            dif, _ = torch.abs(image_point - image_data).max(0)
            plt.imsave('adsfa.png', dif)
            plt.show()


    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class KeypointData(Dataset):
    def __init__(self, basedir, transforms=None):

        files = glob.glob(os.path.join(basedir,'*.csv'))

        self.transforms = transforms
        self.images = []
        self.labels = []
        self.boxes = []
        self.keypoints = []

        for f in files:
            df = pandas.read_csv(f)
            image = TF.to_tensor(PIL.Image.open(f[:-4:] +  '.png'))
            image = torch.cat((image, image, image), dim=0)

            boxes = None
            keypoints = None
            labels = []

            for l in range(len(df)):
                d = json.loads(df['region_shape_attributes'][l])
                label = json.loads(df['region_attributes'][l])
                try:
                    labels.append(int(label['stereocillia']))
                except KeyError:
                    labels.append(1)

                if d['name'] == 'rect':
                    x = d['x']
                    y = d['y']
                    width = d['width']
                    height = d['height']
                    box = torch.tensor([x, y, x + width, y + height])
                    # Basically just append boxes to end of boxes
                    if boxes is None:
                        boxes = box.unsqueeze(0)
                    else:
                        boxes = torch.cat((boxes, box.unsqueeze(0)), dim=0)

                if d['name'] == 'polyline':
                    x = torch.tensor(d['all_points_x'])
                    y = torch.tensor(d['all_points_y'])
                    k = torch.ones(x.shape)
                    polyline = torch.cat((x.unsqueeze(-1),y.unsqueeze(-1),k.unsqueeze(-1)),dim=-1)
                    if keypoints is None:
                        keypoints = polyline.unsqueeze(0)
                    else:
                        try:
                            keypoints = torch.cat((keypoints, polyline.unsqueeze(0)), dim=0)
                        except RuntimeError:
                            raise RuntimeError('accidental extra kepoint value. Should only be 3', l, f)


            self.labels.append(torch.tensor(labels))
            self.boxes.append(boxes)
            self.keypoints.append(keypoints)
            self.images.append(image)

    def __getitem__(self, item):
        data_dict ={'boxes': self.boxes[item],
                    'labels': self.labels[item],
                    'keypoints': self.keypoints[item],
                    'mask': self.images[item]}

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        return data_dict['mask'], data_dict

    def __len__(self):
        return len(self.images)
