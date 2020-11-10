import torch
import torchvision.transforms.functional
from typing import List, Dict


class random_v_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate

        example_input = torch.randn((1, 3, 100, 100, 10))
        self.fun = torch.jit.trace(torchvision.transforms.functional.vflip, example_inputs=example_input)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.randn(1) < self.rate:
            data_dict['mask'] = self.fun(data_dict['mask'])
            data_dict['masks'] = self.fun(data_dict['masks'])

        return data_dict


class random_h_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        example_input = torch.randn((1, 3, 100, 100, 10))
        self.fun = torch.jit.trace(torchvision.transforms.functional.hflip, example_inputs=example_input)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.randn(1) < self.rate:
            data_dict['mask'] = self.fun(data_dict['mask'])
            data_dict['masks'] = self.fun(data_dict['masks'])

        return data_dict


class gaussian_blur:
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5) -> None:
        self.kernel_targets = kernel_targets
        self.rate = rate

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.randn(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            data_dict['mask'] = torchvision.transforms.functional.gaussian_blur(data_dict['mask'], kern)
        return data_dict


class random_resize:
    def __init__(self, rate: float = 0.5, scale: tuple = (300, 1440)) -> None:
        self.rate = rate
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly resizes an mask

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.randn(1) < self.rate:
            size = torch.randint(self.scale[0], self.scale[1], (1, 1)).item()
            data_dict['mask'] = torchvision.transforms.functional.resize(data_dict['mask'], size)
            data_dict['masks'] = torchvision.transforms.functional.resize(data_dict['masks'], size)

        return data_dict


class adjust_brightness:
    def __init__(self, rate=.5, range_brightness=(.3, 1.7)):
        self.rate = rate
        self.range = range_brightness

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        example_input = torch.randn((1, 3, 100, 100, 10)).to(device)
        self.fun = torch.jit.trace(torchvision.transforms.functional.adjust_brightness,
                                   example_inputs=(example_input, torch.tensor([0.5]).to(device)))

    def __call__(self, input):
        image = input['mask']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1]).to(image.device)
        if torch.randn(1) < self.rate:
            image = self.fun(image, val)

        return {'mask': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class adjust_contrast:
    def __init__(self, rate=.5, range_contrast=(.3, 1.7)):
        self.rate = rate
        self.range = range_contrast

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        example_input = torch.randn((1, 3, 100, 100, 10)).to(device)
        self.fun = torch.jit.trace(torchvision.transforms.functional.adjust_brightness,
                                   example_inputs=(example_input, torch.tensor([0.5]).to(device)))

    def __call__(self, input):
        image = input['mask']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']
        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1]).to(image.device)
        if torch.randn(1) < self.rate:
            image = torchvision.transforms.functional.adjust_contrast(image, val)

        return {'mask': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class random_affine:
    def __init__(self, rate=0.5, angle=(-180, 180), shear=(-45, 45), scale=(0.9, 1.5)):
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

        # img, angle, translate, scale, shear
        example_input = (example_image, angle, translate, scale, shear)
        affine(example_image, angle, translate, scale, shear)
        self.fun = affine  # torch.jit.trace(affine, example_inputs=example_input)
        # self.fun(example_image, angle, translate, scale, shear)

    def __call__(self, input):
        image = input['mask']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1]).to(self.device)
        shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1]).to(self.device)
        scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]).to(self.device)

        translate = torch.tensor([0, 0])
        # (img, angle, translate, scale, shear)
        if torch.randn(1) < self.rate:
            image = self.fun(image, angle, translate, scale, shear)
            masks = self.fun(masks, angle, translate, scale, shear)
            #
            # mask = self.fun(img=mask, angle=angle, shear=shear, scale=scale, translate=translate)
            # masks = self.fun(img=masks, angle=angle, shear=shear, scale=scale, translate=translate)

        return {'mask': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class to_cuda:
    def __init__(self):
        pass

    def __call__(self, input: dict = None) -> dict:
        for key in input:
            input[key] = input[key].cuda()
        return input


class to_tensor:
    def __init__(self):
        pass

    def __call__(self, input):
        image = input['mask']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        image = torchvision.transforms.functional.to_tensor(image)

        return {'mask': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


class correct_boxes:
    def __init__(self):
        pass

    def __call__(self, input):
        return _correct_box(image=input['mask'], masks=input['masks'], labels=input['labels'])


class stack_image:
    def __init__(self):
        pass

    def __call__(self, input):
        image = input['mask']
        boxes = input['boxes']
        masks = input['masks']
        labels = input['labels']

        image = torch.cat((image, image, image), dim=0)

        return {'mask': image, 'masks': masks, 'boxes': boxes, 'labels': labels}


@torch.jit.script
def get_box_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Returns the bounding box for a particular segmentation mask
    :param: mask torch.Tensor[X,Y] some mask where 0 is background and !0 is a segmentation mask
    :return: torch.Tensor[4] coordinates of the box surrounding the segmentation mask [x1, y1, x2, y2]
    """
    ind = torch.nonzero(mask)

    if ind.shape[0] == 0:
        box = torch.tensor([0, 0, 0, 0])

    else:
        box = torch.empty(4).to(mask.device)
        x = ind[:, 1]
        y = ind[:, 0]
        torch.stack((torch.min(x), torch.min(y), torch.max(x), torch.max(y)), out=box)

    return box


@torch.jit.script
def _correct_box(image: torch.Tensor,  masks: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:

    boxes = torch.cat([get_box_from_mask(m).unsqueeze(0) for m in masks], dim=0)
    ind = torch.tensor([m.max().item() > 0 for m in masks], dtype=torch.bool)

    return {'mask': image, 'masks': masks[ind, :, :], 'boxes': boxes[ind, :], 'labels': labels[ind]}



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
