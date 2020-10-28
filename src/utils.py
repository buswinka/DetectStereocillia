import  matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import PIL.Image
import skimage.io
import skimage.color


def show_box_pred_simple(image, boxes):

    c = ['nul','r','b','y','w']

    # x1, y1, x2, y2


    plt.imshow(image)
    plt.tight_layout()

    for i, box in enumerate(boxes):

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1,x2],[y2,y2],'r', lw=0.5)
        plt.plot([x1,x2],[y1,y1],'r', lw=0.5)
        plt.plot([x1,x1],[y1,y2],'r', lw=0.5)
        plt.plot([x2,x2],[y1,y2],'r', lw=0.5)
    plt.show()


def render_mask(image: torch.Tensor, model_output:dict, threshold:float) -> None:

    image = image[0,0,:,:].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()
    masks = model_output['masks'].detach().cpu().numpy() > 0.5

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    colormask = np.zeros((masks.shape[-2], masks.shape[-1], 3))

    simple_colors = [[0,0,1],[0,1,0],[1,0,0] ]

    for i, box in enumerate(boxes):
        try:
            if model_output['scores'][i] < threshold:
                continue
        except KeyError:
            pass

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1,x2],[y2,y2],'r', lw=0.5)
        plt.plot([x1,x2],[y1,y1],'r', lw=0.5)
        plt.plot([x1,x1],[y1,y2],'r', lw=0.5)
        plt.plot([x2,x2],[y1,y2],'r', lw=0.5)

        if masks.ndim == 4:
            colormask[:,:,0][masks[i,0, :, :]>0] = simple_colors[model_output['labels'][i] - 1][0]
            colormask[:,:,1][masks[i,0, :, :]>0] = simple_colors[model_output['labels'][i] - 1][1]
            colormask[:,:,2][masks[i,0, :, :]>0] = simple_colors[model_output['labels'][i] - 1][2]
        else:
            colormask[:, :, 0][masks[i, :, :] > 0] = simple_colors[model_output['labels'][i] - 1][0]
            colormask[:, :, 1][masks[i, :, :] > 0] = simple_colors[model_output['labels'][i] - 1][1]
            colormask[:, :, 2][masks[i, :, :] > 0] = simple_colors[model_output['labels'][i] - 1][2]
    plt.imshow(colormask, alpha=0.35)
    plt.show()


def render_boxes(image: torch.Tensor, model_output:dict, threshold: float) -> None:

    image = image[0,0,:,:].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    for i, box in enumerate(boxes):
        if model_output['scores'][i] < threshold:
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1,x2],[y2,y2],'r', lw=0.5)
        plt.plot([x1,x2],[y1,y1],'r', lw=0.5)
        plt.plot([x1,x1],[y1,y2],'r', lw=0.5)
        plt.plot([x2,x2],[y1,y2],'r', lw=0.5)

    plt.show()

class image:
    def __init__(self, base_im):
        self.image = base_im
        self.colormask = np.zeros((3, self.image.shape[-2], self.image.shape[-1]))
        self.simple_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    def add_partial_maks(self, x:int, y:int, model_output:dict, threshold:float):

        masks = model_output['masks'].detach().cpu().numpy()

        for i in range(masks.shape[0]):
            try:
                if model_output['scores'][i] < threshold:
                    continue
            except KeyError:
                pass

            if masks.ndim == 4:
                self.colormask[0,x:x+masks.shape[-2], y:y+masks.shape[-1]][masks[i, 0, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][0]
                self.colormask[1,x:x+masks.shape[-2], y:y+masks.shape[-1]][masks[i, 0, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][1]
                self.colormask[2,x:x+masks.shape[-2], y:y+masks.shape[-1]][masks[i, 0, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][2]
            else:
                self.colormask[x:x+masks.shape[1]-1, y:y+masks.shape[2]-1, 0][masks[i, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][0]
                self.colormask[x:x+masks.shape[1]-1, y:y+masks.shape[2]-1, 1][masks[i, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][1]
                self.colormask[x:x+masks.shape[1]-1, y:y+masks.shape[2]-1, 2][masks[i, :, :] > 0.5] = self.simple_colors[model_output['labels'][i] - 1][2]

    def render(self, filename='test.png'):
        # plt.figure(figsize=(10,10))
        # plt.imshow(self.image[0,0,:,:], cmap='Greys_r')
        # plt.tight_layout()
        # plt.imshow(self.colormask, alpha=0.35)
        # plt.savefig('test.png',dpi=300)
        # plt.show()
        print(self.image[0,0,:,:].shape)
        im = self.image.numpy()[0,:,:,:].astype(np.float)
        cm = self.colormask.astype(np.float)
        out = cv2.addWeighted(cm, 0.35, im, 1-0.35, 0)
        out = skimage.img_as_uint(out)
        skimage.io.imsave(filename, out.transpose((1,2,0)))
        print(out.shape, out.dtype)




def color_from_ind(i):
    """
    Take in some number and always generate a unique color from that number.
    Quick AF
    :param i:
    :return:
    """
    np.random.seed(i)
    return np.random.random(3)

