import os.path
import os
import skimage.io as io
import torchvision.transforms.functional as TF
import PIL
import torch

import src.utils

def save(data, image_path):
    image_folder_path = os.path.splitext(image_path)[0]
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)

    image = TF.to_tensor(PIL.Image.open(image_path))
    image = torch.cat((image, image, image), dim=0)

    masks = data['masks'].detach().cpu()
    scores = data['scores'].detach().cpu()
    labels = data['labels'].detach().cpu()

    image_renderer = src.utils.image(image)
    image_renderer.add_partial_maks(x=0, y=0, model_output=data, threshold=0.5)
    image_renderer.render(os.path.join(image_folder_path, 'masked_image.png'))

    f = open(os.path.join(image_folder_path, 'data.csv'),'w')
    f.write('num,row,score,filename\n')

    for i in range(masks.shape[0]):
        m = masks[i, :, :]
        score = scores[i]
        cell = 'row' + str(4-int(labels[i].item()))
        io.imsave(os.path.join(image_folder_path, f'mask_{cell}_{i}_{score}.tif'), m.numpy())
        f.write(f'{i},{str(4-int(labels[i].item()))},{score},mask_{cell}_{i}_{score}.tif\n')


    f.close()