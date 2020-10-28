import  matplotlib.pyplot as plt
import numpy as np
import torch
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


def render_mask(image: torch.Tensor, model_output:dict) -> None:

    image = image[0,0,:,:].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()
    masks = model_output['masks'].detach().cpu().numpy() > 0.5

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    colormask = np.zeros((masks.shape[-2], masks.shape[-1], 3))

    simple_colors = [[0,0,1],[0,1,0],[1,0,0] ]

    for i, box in enumerate(boxes):
        try:
            if model_output['scores'][i] < 0.3:
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


def render_boxes(image: torch.Tensor, model_output:dict) -> None:

    image = image[0,0,:,:].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    for i, box in enumerate(boxes):
        if model_output['scores'][i] < 0.3:
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


def color_from_ind(i):
    """
    Take in some number and always generate a unique color from that number.
    Quick AF
    :param i:
    :return:
    """
    np.random.seed(i)
    return np.random.random(3)



def color_from_ind(i):
    """
    Take in some number and always generate a unique color from that number.
    Quick AF
    :param i:
    :return:
    """
    np.random.seed(i)
    return np.random.random(3)

