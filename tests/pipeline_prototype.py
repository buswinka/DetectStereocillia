from src.model import faster_rcnn, mask_rcnn
from src.dataloader import BundleData, MaskData
import torch
import src.utils
import torch.optim
import matplotlib.pyplot as plt

faster_rcnn.load_state_dict(torch.load('./faster_rcnn.mdl'))
faster_rcnn.eval().cuda()

mask_rcnn.load_state_dict(torch.load('./mask_rcnn.mdl'))
mask_rcnn.eval().cuda()

tests = BundleData('../data/bundle_train/')

image, _ = tests[1]

with torch.no_grad():
    bundles = faster_rcnn(image.cuda())[0]
    for box in bundles['boxes']:
        slice = image[:, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        masks = mask_rcnn(slice.cuda())[0]
        src.utils.render_mask(slice, masks)






