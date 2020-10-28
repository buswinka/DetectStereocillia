from src.model import faster_rcnn, mask_rcnn
from src.dataloader import BundleData, MaskData
import torch
import src.utils
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import src.transforms as t
import torchvision.transforms

faster_rcnn.load_state_dict(torch.load('./faster_rcnn.mdl'))
faster_rcnn.eval().cuda()

mask_rcnn.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/tests/mask_rcnn.mdl'))
mask_rcnn.eval().cuda()

tests = BundleData('../data/bundle_train/')

image, _ = tests[1]
transforms = torchvision.transforms.Compose([
    t.random_h_flip(),
    t.random_v_flip(),
    t.random_affine(),
    t.adjust_brightness(),
    t.adjust_contrast(),
    t.correct_boxes()
])

tests = MaskData('../data/train/', transforms=transforms)
# tests = DataLoader(tests, batch_size=None, shuffle=True, num_workers=4)

larger_boi = src.utils.image(image.unsqueeze(0))

with torch.no_grad():
    bundles = faster_rcnn(image.unsqueeze(0).cuda())[0]
    for box in bundles['boxes']:
        slice = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        masks = mask_rcnn(slice.unsqueeze(0).cuda())[0]
        # src.utils.render_mask(slice.unsqueeze(0), masks, 0.0)
        larger_boi.add_partial_maks(x=int(box[1]), y=int(box[0]), model_output=masks, threshold=0.10)

larger_boi.render()


image, _ = tests[0]
with torch.no_grad():
    masks = mask_rcnn(image.unsqueeze(0).cuda())[0]
    src.utils.render_mask(image.unsqueeze(0), masks, 0.0)








