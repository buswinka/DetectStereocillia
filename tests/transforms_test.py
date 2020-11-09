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
    x = torch.randn((1,3,5000,5000))
    gb = src.transforms.gaussian_blur()
    gb(x)
