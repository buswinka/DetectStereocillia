from src.model import faster_rcnn
from src.dataloader import BundleData
import src.utils
import torch.optim
import matplotlib.pyplot as plt

tests = BundleData('../data/bundle_train/')

faster_rcnn = faster_rcnn.train().cuda()

optimizer = torch.optim.Adam(faster_rcnn.parameters(), lr = 0.0001)

for epoch in range(300):
    for image, data in tests:
        for key in data:
            data[key] = data[key].cuda()

        optimizer.zero_grad()
        loss = faster_rcnn(image.cuda(), [data])
        losses = 0
        for key in loss:
            losses += loss[key]
        losses.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(epoch, losses)

faster_rcnn.eval()
out = faster_rcnn(image.cuda())
out = out[0]

src.utils.render_boxes(image, out)
