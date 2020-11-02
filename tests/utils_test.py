import src.utils
from src.dataloader import KeypointData
import src.model

import torch



def test_render_keypoints():
    model = src.model.keypoint_rcnn

    device = 'cuda'

    data = KeypointData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/keypoint_train_data')
    model.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/models/keypoint_rcnn.mdl'))
    model = model.eval().to(device)

    image, _ = data[0]
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))

    src.utils.render_keypoints(image.unsqueeze(0), out[0], 0.0)