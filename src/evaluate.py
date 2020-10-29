from src.model import faster_rcnn, mask_rcnn
from src.dataloader import BundleData, MaskData
import torch
import src.utils
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import src.transforms as t
import torchvision.transforms
import torchvision.transforms.functional as TF
import PIL

import os.path

def evaluate(eval_path):

    if torch.cuda.is_available(): device = 'cuda:0'
    else: device = 'cpu'

    base_dir, filename = os.path.split(eval_path)
    base_file_name = os.path.splitext(filename)[0]

    models_path = os.path.join(os.getcwd(), 'models')

    faster_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'faster_rcnn.mdl')))
    faster_rcnn.eval().to(device)
    mask_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'mask_rcnn.mdl')))
    mask_rcnn.eval().to(device)

    image = TF.to_tensor(PIL.Image.open(eval_path))
    image = torch.cat((image, image, image), dim=0)

    larger_boi = src.utils.image(image.unsqueeze(0))

    masks = mask_rcnn(image.unsqueeze(0).to(device))[0]
    larger_boi.add_partial_maks(x=0, y=0, model_output=masks, threshold=0.25)
    larger_boi.render(os.path.join(base_dir, base_file_name+'_mask.png'))











