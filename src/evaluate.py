from src.model import faster_rcnn, mask_rcnn, keypoint_rcnn
from src.dataloader import BundleData, MaskData, KeypointData
import torch
import src.utils
import torch.optim
import torchvision.ops as ops
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

            keypoint_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'keypoint_rcnn.mdl')))
            keypoint_rcnn.eval().to(device)

            self.mask_rcnn = mask_rcnn
            self.keypoint_rcnn = keypoint_rcnn

        except FileNotFoundError:
            cwd = os.getcwd()
            os.chdir('..')
            models_path = os.path.join(os.getcwd(), 'models')
            os.chdir(cwd)
            mask_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'mask_rcnn.mdl')))
            mask_rcnn.eval().to(device)

            keypoint_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'keypoint_rcnn.mdl')))
            keypoint_rcnn.eval().to(device)

            self.mask_rcnn = mask_rcnn
            self.keypoint_rcnn = keypoint_rcnn

    def __call__(self, eval_path):

        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'

        image = TF.to_tensor(PIL.Image.open(eval_path))
        image = torch.cat((image, image, image), dim=0)
        larger_boi = src.utils.image(image.unsqueeze(0))

        with torch.no_grad():
            masks = self.mask_rcnn(image.unsqueeze(0).to(device))[0]
            keypoints = self.keypoint_rcnn(image.unsqueeze(0).to(device))[0]

        index = ops.nms(masks['boxes'], masks['scores'], 0.5)

        masks['masks']=masks['masks'][index,:,:]
        masks['scores']=masks['scores'][index]
        masks['labels']=masks['labels'][index]
        masks['boxes']=masks['boxes'][index, :]

        larger_boi.add_partial_maks(x=0, y=0, model_output=masks, threshold=0.50)

        return larger_boi.render_mat(), masks, keypoints











