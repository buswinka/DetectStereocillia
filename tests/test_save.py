from src.model import faster_rcnn, mask_rcnn
from src.dataloader import BundleData, MaskData
import src.transforms as t
import src.utils
import src.save

import torch
import torch.optim
import torchvision.transforms
import torch.optim
import torchvision.transforms
import torchvision.transforms.functional as TF
import PIL

def test_save_mask():
    faster_rcnn.load_state_dict(
        torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/models/faster_rcnn.mdl'))
    faster_rcnn.eval().cuda()

    mask_rcnn.load_state_dict(
        torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/models/mask_rcnn.mdl'))
    mask_rcnn.eval().cuda()

    image_path = '/data/test/16k01.TIF'
    image = TF.to_tensor(PIL.Image.open(image_path))
    image = torch.cat((image, image, image), dim=0)

    device='cuda'

    with torch.no_grad():
        data_out = mask_rcnn(image.unsqueeze(0).to(device))[0]
        src.save.save(data_out, image_path)