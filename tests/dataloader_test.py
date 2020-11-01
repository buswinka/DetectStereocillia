from src.model import mask_rcnn
from src.dataloader import MaskData, ChunjieData
import src.transforms as t
import src.utils
import torch.optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def test_MaskData_stress():
    transforms = torchvision.transforms.Compose([
        t.random_h_flip(),
        t.random_v_flip(),
        t.random_affine(),
        t.adjust_brightness(),
        t.adjust_contrast(),
        t.correct_boxes()
    ])

    tests = MaskData('./data/train/', transforms=transforms)
    tests = DataLoader(tests, batch_size=None, num_workers=4)
    for i in range(10):
        for image, data in tests:
            if image.max() == 0:
                raise ValueError('Donefuckedup')
            if data['boxes'].shape[1] != 4:
                raise ValueError(data['boxes'].shape)


            for i, box in enumerate(data['boxes']):
                if box.sum()==0:
                    raise ValueError('Doestn work')

    return None

def test_ChunjieData_stress():
    data = ChunjieData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/chunjie_identification_train/')

    return None

test_ChunjieData_stress()

