import src.model
from src.dataloader import MaskData, KeypointData, FasterRCNNData
import src.transforms as t
import src.utils
import torch.optim

import numpy as np
import time
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image

path = '/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train/faster'
epochs = 1

transforms = torchvision.transforms.Compose([
    t.to_cuda(),
    t.random_resize(scale=(250, 800)),
    t.normalize(),
    t.random_h_flip(),
    t.random_v_flip(),
    t.gaussian_blur(kernel_targets=torch.tensor([3, 5, 7])),
    t.random_affine(),
    t.stack_image(),
    t.adjust_brightness(),
    t.adjust_contrast(),
    t.correct_boxes(),
])

data = FasterRCNNData(path, transforms)

transforms = torchvision.transforms.Compose([
    t.to_cuda(),
    t.normalize(),
    t.correct_boxes()
])

val_path = '/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/validate/faster'
validate = FasterRCNNData(val_path, transforms)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = src.model.faster_rcnn_cilia
model.load_state_dict(torch.load('../models/faster_rcnn_cilia_Pretrained.mdl'))
model.train().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

out_str = ''
if len(data) < 1:
    raise ValueError('No available train data')
for e in range(epochs):
    epoch_loss = []
    val_loss = []
    time_1 = time.clock_gettime_ns(1)
    for image, data_dict in data:
        optimizer.zero_grad()
        loss = model(image.unsqueeze(0), [data_dict])
        losses = 0
        for key in loss:
            losses += loss[key]
        losses.backward()
        epoch_loss.append(losses.item())
        optimizer.step()
        del image, data_dict, losses

    for image, data_dict in validate:
        with torch.no_grad():
            loss = model(image.unsqueeze(0), [data_dict])
            losses = 0
            for key in loss:
                losses += loss[key]
            val_loss.append(losses.item())

        del image, data_dict, losses




    time_2 = time.clock_gettime_ns(1)
    delta_time = (np.abs(time_2 - time_1) / 1e9) / 60

    #  --------- This is purely to output a nice bar for training --------- #
    if e % 1 == 0:
        if e > 0:
            print('\b \b' * len(out_str), end='', flush=True)
        progress_bar = '[' + '█' * +int(np.round(e / epochs, decimals=1) * 10) + \
                       ' ' * int((10 - np.round(e / epochs, decimals=1) * 10)) + f'] {np.round(e / epochs, decimals=3)}% '

        out_str = f'epoch: {e}/{epochs} ' + progress_bar + f'| time remaining: {delta_time * (epochs - e)} ' \
                                                          f'| epoch loss: {torch.tensor(epoch_loss).mean().item()}' \
                                                          f' validation loss: {torch.tensor(val_loss).mean().item()}'
        print(out_str, end='', flush=True)

    # If its the final epoch print out final string
    elif e == epochs - 1:
        print('\b \b' * len(out_str), end='')
        progress_bar = '[' + '█' * 10 + f'] {1.0}'
        out_str = f'epoch: {epochs}/{epochs} ' + progress_bar + f'| time remaining: {0} ' \
                                                          f'| epoch loss: {torch.tensor(epoch_loss).mean().item()}' \
                                                                f' val loss: {torch.tensor(val_loss).mean().item()}'
        print(out_str)

torch.save(model.state_dict(), '../models/faster_rcnn_cilia.mdl')


# ___________ CHECK ___________ #

image = TF.to_tensor(Image.open('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train/faster/16 K 16.png')).pin_memory()
im_unnorm = t.stack_image()({'image': image})['image']
im = t.stack_image()(t.normalize()({'image': image}))['image']
model.eval()
with torch.no_grad():
    out = model(im.unsqueeze(0).cuda())[0]
image = image.cpu().numpy()
boxes = out['boxes'].detach().cpu().numpy()
c = ['nul','r','b','y','w']
plt.imshow(im_unnorm[0, :, :], cmap='Greys_r')
plt.tight_layout()
for i, box in enumerate(boxes):
    if out['scores'][i] < 0.5:
        continue

    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    l = out['labels'][i].detach().cpu().numpy()

    plt.plot([x1,x2],[y2,y2],c='C'+str(l), lw=1)
    plt.plot([x1,x2],[y1,y1],c='C'+str(l), lw=1)
    plt.plot([x1,x1],[y1,y2],c='C'+str(l), lw=1)
    plt.plot([x2,x2],[y1,y2],c='C'+str(l), lw=1)

plt.show()