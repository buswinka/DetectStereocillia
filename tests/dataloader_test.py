from src.model import mask_rcnn
from src.dataloader import MaskData, ChunjieData, KeypointData
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

    print(mask.shape)
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


test_MaskData_stress()

