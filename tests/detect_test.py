from src.model import faster_rcnn, keypoint_rcnn
from src.dataloader import BundleData, KeypointData
import src.utils
import torch.optim
import matplotlib.pyplot as plt


def test_bundle_detection_scheme():
    tests = BundleData('./data/bundle_train/')
    assert len(tests) > 0

    fast = faster_rcnn
    fast = fast.train().cuda()

    optimizer = torch.optim.Adam(faster_rcnn.parameters(), lr = 0.0001)

    image = []

    for epoch in range(10):
        for image, data in tests:
            for key in data:
                data[key] = data[key].cuda()

            optimizer.zero_grad()
            loss = fast(image.unsqueeze(0).cuda(), [data])
            losses = 0
            for key in loss:
                losses += loss[key]
            losses.backward()
            optimizer.step()

    fast.eval()
    out = faster_rcnn(image.unsqueeze(0).cuda())
    # torch.save(faster_rcnn.state_dict(), '/src/faster_rcnn.mdl')
    # src.utils.render_boxes(image.unsqueeze(0), out, 0.5)
    assert True


def test_keypoint_model():
    tests = KeypointData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/keypoint_train_data')

    model = keypoint_rcnn.train().cuda()
    optimizer = torch.optim.Adam(faster_rcnn.parameters(), lr=0.0001)


    for epoch in range(10):
        for image, data in tests:
            for key in data:
                data[key] = data[key].cuda()

            optimizer.zero_grad()
            loss = model(image.unsqueeze(0).cuda(), [data])
            losses = 0
            for key in loss:
                losses += loss[key]
            losses.backward()
            optimizer.step()

    model.eval()
    out = model(image.unsqueeze(0).cuda())