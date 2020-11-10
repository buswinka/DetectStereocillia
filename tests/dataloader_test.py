from src.model import mask_rcnn
from src.dataloader import MaskData, KeypointData, FasterRCNNData
import src.transforms as t
import src.utils
import torch.optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def test_MaskData_stress():
    return True
    transforms = torchvision.transforms.Compose([
        t.to_cuda(),
        t.gaussian_blur(),
        t.random_h_flip(),
        t.random_v_flip(),
        t.random_affine(),
        t.adjust_brightness(),
        t.adjust_contrast(),
        t.correct_boxes()
    ])

    tests = MaskData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train', transforms=transforms)
    # tests = DataLoader(tests, batch_size=None, num_workers=4)

    image, data_dict = tests[1]
    mask = data_dict['masks']

    m,_ = mask.max(dim=0)
    plt.imshow(m)
    plt.show()

    plt.imshow(image[0,:,:])
    plt.show()

    return None

def test_ChunjieData_stress():
    return True
    transforms = [src.transforms.to_cuda]
    data = ChunjieData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/chunjie_identification_train/',
                       transforms)

    return None

def test_keypoint_dataloader():
    return True
    data = KeypointData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/keypoint_train_data')
    image, data_dict = data[0]

    assert data_dict['boxes'].shape[1] == 4
    assert data_dict['boxes'].shape[0] == data_dict['keypoints'].shape[0]
    assert data_dict['keypoints'].shape[2] == 3


def test_fasterrcnn_dataloader():
    path = '/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train/faster'
    transforms = torchvision.transforms.Compose([
        t.to_cuda(),
        t.stack_image(),
        t.correct_boxes()
    ])

    data = FasterRCNNData(path, transforms)
    image, data_dict = data[0]

    boxes = data_dict['boxes']

    image = image.cpu().numpy()
    boxes = data_dict['boxes'].detach().cpu().numpy()

    c = ['nul', 'r', 'b', 'y', 'w']

    # x1, y1, x2, y2

    plt.imshow(image[0, :, :],cmap='Greys_r')

    plt.tight_layout()
    print(data_dict['labels'])

    for i, box in enumerate(boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        l = data_dict['labels'][i].detach().cpu().numpy()

        plt.plot([x1, x2], [y2, y2], c='C' + str(l), lw=0.5)
        plt.plot([x1, x2], [y1, y1], c='C' + str(l), lw=0.5)
        plt.plot([x1, x1], [y1, y2], c='C' + str(l), lw=0.5)
        plt.plot([x2, x2], [y1, y2], c='C' + str(l), lw=0.5)

    plt.show()

test_fasterrcnn_dataloader()
