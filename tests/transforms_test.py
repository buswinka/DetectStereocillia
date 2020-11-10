import torch
import src.transforms
import numpy

def test_get_box_from_mask():
    mask = torch.eye(100)
    mask[0,0]=0
    mask[-1,-1]=0
    out = src.transforms.get_box_from_mask(mask)
    assert out is not None


def test_gaussian_blur():
    data = {'mask': torch.randn((1,3,50,50)), 'masks':None, 'boxes':None, 'labels':None}
    gb = src.transforms.gaussian_blur()
    gb(data)

