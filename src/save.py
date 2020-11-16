import os.path
import os
import skimage.io as io
import torchvision.transforms.functional as TF
import PIL
import torch

import src.utils

def save_mask(data, image_path):
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

    f = open(os.path.join(image_folder_path, 'mask_data.csv'),'w')
    f.write('num,row,score,save_name\n')

    for i in range(masks.shape[0]):
        m = masks[i, :, :]
        score = scores[i]
        cell = 'row' + str(4-int(labels[i].item()))
        io.imsave(os.path.join(image_folder_path, f'mask_{cell}_{i}_{score}.tif'), m.numpy())
        f.write(f'{i},{str(4-int(labels[i].item()))},{score},mask_{cell}_{i}_{score}.tif\n')


    f.close()


def save_boxes(data, image_path):
    image_folder_path = os.path.splitext(image_path)[0]
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)


    boxes = data['boxes'].detach().cpu()
    scores = data['scores'].detach().cpu()
    labels = data['labels'].detach().cpu()

    f = open(os.path.join(image_folder_path, 'box_data.csv'),'w')
    f.write('num\tlabel\tscore\tbox\n')

    label = ['Tall+', 'Tall-', 'Mid+', 'Mid-', 'Short+', 'Short-', 'Junk']

    for j in range(boxes.shape[0]):
        i = int(labels[j].item())
        f.write(f'{j}\t{label[i-1]}\t{scores[j]}\t{boxes[j,:].tolist()}\n')

    f.close()