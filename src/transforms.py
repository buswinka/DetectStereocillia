import torch
import torchvision.transforms.functional
import skimage.transform as transform
import numpy as np


class random_v_flip:
    def __init__(self, rate=.5):
        self.rate = rate
        example_input = torch.randn((1,3,100,100,10))
        self.fun = torch.jit.trace(torchvision.transforms.functional.vflip, example_inputs=example_input)

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
            image = self.fun(image)
            masks = self.fun(masks)

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


class gaussian_blur:
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5):
        self.kernel_targets = kernel_targets
        self.rate = rate

    def __call__(self, input):
        if torch.randn(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            image = torchvision.transforms.functional.gaussian_blur(input['image'], kern)
        else:
            image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class random_resize:
    def __init__(self, rate=.5, scale=(300, 1440)):
        self.rate = rate
        self.scale = scale

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        if torch.randn(1) < self.rate:
            size = torch.randint(self.scale[0], self.scale[1], (1, 1)).item()
            image = torchvision.transforms.functional.resize(image, size)
            masks = torchvision.transforms.functional.resize(masks, size)

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}



class adjust_brightness:
    def __init__(self, rate=.5, range_brightness=(.3,1.7)):
        self.rate = rate
        self.range = range_brightness

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        example_input = torch.randn((1,3,100,100,10)).to(device)
        self.fun = torch.jit.trace(torchvision.transforms.functional.adjust_brightness,
                                   example_inputs=(example_input, torch.tensor([0.5]).to(device)))

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1]).to(image.device)
        if torch.randn(1) < self.rate:
            image = self.fun(image, val)

        return {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}


class adjust_contrast:
    def __init__(self, rate=.5, range_contrast=(.3,1.7)):
        self.rate = rate
        self.range = range_contrast

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        example_input = torch.randn((1, 3, 100, 100, 10)).to(device)
        self.fun = torch.jit.trace(torchvision.transforms.functional.adjust_brightness,
                                   example_inputs=(example_input, torch.tensor([0.5]).to(device)))

    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1]).to(image.device)
        if torch.randn(1) < self.rate:
            image = torchvision.transforms.functional.adjust_contrast(image, val)

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class random_affine:
    def __init__(self, rate=0.5, angle=(-180,180),shear=(-45,45), scale=(0.9, 1.5)):
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        example_image = torch.randn((1, 3, 100, 100)).to(self.device)
        angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1]).to(self.device)
        shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1]).to(self.device)
        scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]).to(self.device)
        translate = torch.tensor([0, 0]).to(self.device)

        #img, angle, translate, scale, shear
        example_input = (example_image, angle, translate, scale, shear)
        affine(example_image, angle, translate,scale,shear)
        self.fun = affine #torch.jit.trace(affine, example_inputs=example_input)
        # self.fun(example_image, angle, translate, scale, shear)

    def __call__(self, input):

        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1]).to(self.device)
        shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1]).to(self.device)
        scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]).to(self.device)

        translate = torch.tensor([0,0])
        # (img, angle, translate, scale, shear)
        if torch.randn(1) < self.rate:
            image = self.fun(image, angle, translate, scale, shear)
            masks = self.fun(masks, angle, translate, scale, shear)
            #
            # image = self.fun(img=image, angle=angle, shear=shear, scale=scale, translate=translate)
            # masks = self.fun(img=masks, angle=angle, shear=shear, scale=scale, translate=translate)

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class to_cuda:
    def __init__(self):
        pass
    def __call__(self, input:dict=None) -> dict:
        for key in input:
            input[key] = input[key].cuda()
        return input


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
        self.fun = _correct_box


    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        return self.fun(image, boxes, masks, labels) # {'image':image, 'masks':masks, 'boxes':boxes, 'labels':labels}


class stack_image:
    def __init__(self):
        pass
    def __call__(self, input):
        image = input['image']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        image = torch.cat((image, image, image), dim=0)

        return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


@torch.jit.script
def get_box_from_mask(image: torch.Tensor) -> torch.Tensor:
    ind = torch.nonzero(image)
    if len(ind) == 0:
        return torch.tensor([0,0,0,0])

    box = torch.empty(4).to(image.device)
    x = ind[:, 1]
    y = ind[:, 0]
    torch.stack((torch.min(x),torch.min(y),torch.max(x),torch.max(y)), out=box)
    return box


@torch.jit.script
def _correct_box(image, boxes, masks, labels):
    for i in range(masks.shape[0]):
        if i == 0:
            boxes = get_box_from_mask(masks[i, :, :]).unsqueeze(0).to(image.device)
        else:
            boxes = torch.cat((boxes, get_box_from_mask(masks[i, :, :]).unsqueeze(0).to(image.device)), dim=0)

    ind = torch.ones(masks.shape[0]).long()
    for i in range(len(ind)):
        area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        if not masks[i, :, :].max() or area < 1:  # true or false
            ind[i] = 0

    masks = masks[ind > 0, :, :]
    boxes = boxes[ind > 0, :]
    labels = labels[ind > 0]
    return {'image': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class nul_transform:
    def __init__(self):
        pass
    def __call__(self, input):
        return input


def affine(img, angle, translate, scale, shear):
    angle = float(angle.item())
    scale = float(scale.item())
    shear = float(shear.item())
    translate = translate.tolist()
    return torchvision.transforms.functional.affine(img, angle, translate, scale, shear)
