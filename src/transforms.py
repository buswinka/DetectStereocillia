import torch
import scipy.ndimage as ndimage
import skimage.exposure as exposure
import torchvision.transforms.functional
import skimage.transform as transform
import numpy as np
import copy
import glob
import skimage.io as io
import skimage.morphology
import cv2
import elasticdeform

class random_v_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        """
        Code specifically for transforming images and boxes for fasterRCNN
        Not Compatable with UNET

        :param image:
        :param boxes:
        :return:
        """
        flip = torch.randn(1) < self.rate

        if flip:
            image = torchvision.transforms.functional.vflip(image)
            masks = torchvision.transforms.functional.vflip(masks)

        return {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}


class random_h_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, input):
        """
        FASTER RCNN ONLY

        :param image:
        :param boxes:
        :return:
        """
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        flip = np.random.uniform(0, 1, 1) > 0.5

        if flip:
            image = torchvision.transforms.functional.hflip(image)
            masks = torchvision.transforms.functional.hflip(masks)


        return {'image': image, 'masks': masks, 'boxes':boxes, 'labels':labels}


class random_resize:
    def __init__(self, rate=.5, scale=(.8, 1.2)):
        self.rate = rate
        self.scale = scale

    def __call__(self, image, boxes, masks):
        """
        FASTER RCNN ONLY


        :param image:
        :param boxes:
        :return:
        """

        scale = np.random.uniform(self.scale[0] * 100, self.scale[1] * 100, 1) / 100
        shape = image.shape

        new_shape = np.round(shape * scale)
        new_shape[2] = shape[2]

        image = transform.resize(image, new_shape)

        boxes = np.array(boxes) * scale
        boxes = np.round(boxes).round()
        boxes = np.array(boxes, dtype=np.int64)

        return image, boxes.tolist()


class adjust_brightness:
    def __init__(self, rate=.5, range_brightness=(.5,1.5)):
        self.rate = rate
        self.range = range_brightness

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = float(torch.FloatTensor(1).uniform_(self.range[0], self.range[1]))
        if torch.randn(1) < self.rate:
            image = torchvision.transforms.functional.adjust_brightness(image, val)

        return {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}


class adjust_contrast:
    def __init__(self, rate=.5, range_contrast=(.5,1.5)):
        self.rate = rate
        self.range = range_contrast

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = float(torch.FloatTensor(1).uniform_(self.range[0], self.range[1]))
        if torch.randn(1) < self.rate:
            image = torchvision.transforms.functional.adjust_contrast(image, val)

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}

class random_affine:
    def __init__(self, rate=0.5, angle=(-180,180),shear=(-20,20), scale=(0.9, 1.5)):
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

    def __call__(self, input):

        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        angle = float(torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1]))
        shear = float(torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1]))
        scale = float(torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]))

        if torch.randn(1) < self.rate:
            image = torchvision.transforms.functional.affine(image, angle=angle, shear=shear,
                                                             scale=scale, translate=(0, 0))

            masks = torchvision.transforms.functional.affine(masks, angle=angle, shear=shear,
                                                             scale=scale, translate=(0, 0))

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}

class to_tensor:
    def __init__(self):
        pass

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        image = torchvision.transforms.functional.to_tensor(image)

        return {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}

class correct_boxes:
    def __init__(self):
        pass

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        for i in range(masks.shape[0]):
            if i == 0:
                boxes = get_box_from_mask(masks[i, :, :]).unsqueeze(0)
            else:
                boxes = torch.cat((boxes, get_box_from_mask(masks[i, :, :]).unsqueeze(0)), dim=0)

        ind = torch.ones(masks.shape[0]).long()
        for i in range(len(ind)):
            if masks[i,:,:].max(): #true or false
                ind[i] = 0

        masks = masks[ind, :, :]
        boxes = boxes[ind,:]
        labels = labels[ind]

        return {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}

@torch.jit.script
def get_box_from_mask(image: torch.Tensor) -> torch.Tensor:
    ind = torch.nonzero(image)
    if len(ind) == 0:
        return torch.tensor([0,0,0,0])

    box = torch.empty(4)
    x = ind[:, 1]
    y = ind[:, 0]
    torch.stack((torch.min(x),torch.min(y),torch.max(x),torch.max(y)), out=box)
    return box
