import torch
import src.transforms
import numpy


mask = torch.eye(10000)
mask[0,0]=0
mask[-1,-1]=0

src.transforms.get_box_from_mask(mask)
