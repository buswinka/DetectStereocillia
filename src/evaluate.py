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

class evaluate:
    def __init__(self):

        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'

        try:
            models_path = os.path.join(os.getcwd(), 'models')
            mask_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'mask_rcnn.mdl')))
            mask_rcnn.eval().to(device)

            self.mask_rcnn = mask_rcnn

        except FileNotFoundError:
            cwd = os.getcwd()
            os.chdir('..')
            models_path = os.path.join(os.getcwd(), 'models')
            os.chdir(cwd)
            mask_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'mask_rcnn.mdl')))
            mask_rcnn.eval().to(device)

            self.mask_rcnn = mask_rcnn

    def __call__(self, eval_path):

        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'

        image = TF.to_tensor(PIL.Image.open(eval_path))
        image = torch.cat((image, image, image), dim=0)
        larger_boi = src.utils.image(image.unsqueeze(0))
        masks = self.mask_rcnn(image.unsqueeze(0).to(device))[0]
        larger_boi.add_partial_maks(x=0, y=0, model_output=masks, threshold=0.25)
        return larger_boi.render_mat(), masks











