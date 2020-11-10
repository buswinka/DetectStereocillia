import torch
import matplotlib.pyplot as plt
import os.path
import PIL
import numpy as np
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree
import glob
import pandas
import skimage.io
import skimage.draw
import json
from typing import Optional, Union, Dict, List, Callable
import torchvision.transforms.functional as TF

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class MaskData(Dataset):
    def __init__(self, basedir: str, transforms: Callable, to_cuda: bool = False) -> None:

        files = glob.glob(os.path.join(basedir, 'data.csv'))

        self.transforms = transforms
        self.masks = []
        self.images = []
        self.labels = []
        self.boxes = []

        data_frame = pandas.read_csv(files[0])
        image_names = data_frame['filename'].unique()

        for im_name in image_names:
            df = data_frame[data_frame['filename'] == im_name]
            im_path = os.path.join(basedir, im_name)

            if len(df) <= 1 or not os.path.exists(im_path):  # some dataframes will contain no data... skip
                continue

            image = TF.to_tensor(PIL.Image.open(im_path)).pin_memory()


            mask = torch.zeros((len(df), image.shape[1], image.shape[2]), dtype=torch.uint8)

            region_shape_attributes = df['region_shape_attributes'].to_list()
            region_attributes = df['region_attributes'].to_list()

            # List comprehensions are... beautiful? ;D
            mask = torch.cat([_calculate_mask(json.loads(d), m).unsqueeze(0) for d, m in zip(region_shape_attributes, mask)], dim=0).int()
            boxes = torch.cat([_calculate_box(json.loads(d)).unsqueeze(0) for d in region_shape_attributes], dim=0)
            labels = torch.tensor([_get_label(json.loads(d)) for d in region_attributes])


            for l in range(mask.shape[0]):
                if mask[l, :, :].max() == 0:
                    raise ValueError('Mask is jank')

            self.labels.append(torch.tensor(labels).to_pinned())
            self.masks.append(mask.to_pinned())
            self.boxes.append(boxes.to_pinned())
            self.images.append(image.to_pinned())

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        data_dict ={'boxes': self.boxes[item],
                    'labels': self.labels[item],
                    'masks': self.masks[item],
                    'image': self.images[item]}

        data_dict = self.transforms(data_dict)

        return data_dict['image'], data_dict

    def __len__(self):
        return len(self.images)



class FasterRCNNData(Dataset):
    def __init__(self, basedir: str, transforms: Callable) -> None:

        files = glob.glob(os.path.join(basedir, '*.xml'))

        self.transforms = transforms
        self.masks = []
        self.images = []
        self.labels = []
        self.boxes = []

        for f in files:
            image_path = os.path.splitext(f)[0] + '.png'
            image = TF.to_tensor(PIL.Image.open(image_path)).pin_memory()

            tree = xml.etree.ElementTree.parse(f)
            root = tree.getroot()

            box_from_text = lambda a: [int(a[0].text), int(a[1].text), int(a[2].text), int(a[3].text)]

            im_shape = [image.shape[1], image.shape[2]]

            # Just because you CAN do a list comprehension, doesnt mean you SHOULD
            class_labels = torch.tensor([self._get_class_label(cls.text) for c in root.iter('object') for cls in c.iter('name')])
            bbox_loc = torch.tensor([box_from_text(a) for c in root.iter('object') for a in c.iter('bndbox')])
            mask = torch.cat([self._infer_mask_from_box(b, im_shape).unsqueeze(0) for b in bbox_loc], dim=0)

            self.images.append(image)
            self.boxes.append(bbox_loc.pin_memory())
            self.labels.append(class_labels.pin_memory())
            self.masks.append(mask)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        data_dict ={'boxes': self.boxes[item].clone(),
                    'labels': self.labels[item].clone(),
                    'masks': self.masks[item].clone(),
                    'image': self.images[item].clone()}

        data_dict = self.transforms(data_dict)

        return data_dict['image'], data_dict

    @staticmethod
    def _get_class_label(label_text:str) -> int:
        label = 0
        if label_text == 'Tall+':
            label = 1
        elif label_text == 'Tall-':
            label = 2
        elif label_text == 'Mid+':
            label = 3
        elif label_text == 'Mid-':
            label = 4
        elif label_text == 'Short+':
            label = 5
        elif label_text == 'Short-':
            label = 6
        elif label_text == 'Junk' or label_text == 'junk':
            label = 7
        else:
            raise ValueError(f'Unidentified Label in XML file {label_text}')

        return label

    @staticmethod
    def _infer_mask_from_box(box: torch.Tensor, shape: Union[list, tuple, torch.Tensor]) -> torch.Tensor:
        mask = torch.zeros(shape)
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        mask[y1:y2, x1:x2] = 1
        return mask


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
            image = torch.cat((image, image, image), dim=0).pin_memory()

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


def _calculate_mask(d: Dict[str, list], mask: torch.Tensor) -> torch.Tensor:
    x = d['all_points_x']
    y = d['all_points_y']
    xx, yy = skimage.draw.polygon(x, y)
    mask[yy, xx] = 1
    return mask


def _calculate_box(d: Dict[str, list]) -> torch.Tensor:
    x = d['all_points_x']
    y = d['all_points_y']
    return torch.tensor([np.min(x), np.min(y), np.max(x), np.max(y)])


def _get_label(d: Dict[str, list]) -> torch.Tensor:
    if 'stereocillia' in d:
        label = int(d['stereocillia'])
    elif 'stereocilia' in d:
        label = int(d['stereocilia'])
    else:
        label = None
    return label
