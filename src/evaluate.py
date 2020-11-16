from src.model import faster_rcnn_bundle, faster_rcnn_cilia, mask_rcnn, keypoint_rcnn
import torch
import src.utils
import torch.optim
import torchvision.ops as ops
import src.transforms as t
import torchvision.transforms.functional as TF

import numpy as np

import PIL
from typing import Dict, List, Tuple

import os.path

class evaluate:
    def __init__(self):

        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'

        models_path = os.path.join(os.getcwd(), 'models')
        mask_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'mask_rcnn.mdl')))
        mask_rcnn.eval().to(device)

        faster_rcnn_cilia.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/models/faster_rcnn_cilia_missing.mdl'))
        faster_rcnn_cilia.eval().to(device)

        keypoint_rcnn.load_state_dict(torch.load(os.path.join(models_path, 'keypoint_rcnn.mdl')))
        keypoint_rcnn.eval().to(device)

        self.mask_rcnn = mask_rcnn
        self.keypoint_rcnn = keypoint_rcnn
        self.faster_rcnn_cilia = faster_rcnn_cilia


    def __call__(self, eval_path: str) \
            -> Tuple[np.ndarray, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Evaluates all models on image specified by eval_path

        :param eval_path:
        :return: Tuple[np.ndarray, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            -np.ndarray
        """

        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'

        image = TF.to_tensor(PIL.Image.open(eval_path))
        im_ = t.stack_image()(t.normalize()({'image': image}))['image']
        image = torch.cat((image, image, image), dim=0)
        larger_boi = src.utils.image(image.unsqueeze(0))

        with torch.no_grad():
            masks = self.mask_rcnn(image.unsqueeze(0).to(device))[0]
            keypoints = self.keypoint_rcnn(image.unsqueeze(0).to(device))[0]
            boxes = self.faster_rcnn_cilia(im_.unsqueeze(0).to(device))[0]

        index = ops.nms(masks['boxes'], masks['scores'], 0.5)
        masks['masks'] = masks['masks'][index, :, :]
        masks['scores'] = masks['scores'][index]
        masks['labels'] = masks['labels'][index]
        masks['boxes'] = masks['boxes'][index, :]

        index = ops.nms(boxes['boxes'], boxes['scores'], 0.35)
        boxes['scores'] = boxes['scores'][index]
        boxes['labels'] = boxes['labels'][index]
        boxes['boxes'] = boxes['boxes'][index, :]

        larger_boi.add_partial_maks(x=0, y=0, model_output=masks, threshold=0.50)

        return larger_boi.render_mat(), masks, keypoints, boxes
