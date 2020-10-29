from src.model import faster_rcnn
from src.dataloader import BundleData
import src.utils
import torch.optim
import matplotlib.pyplot as plt

tests = BundleData('../data/bundle_train/')

faster_rcnn = faster_rcnn.train().cuda()

optimizer = torch.optim.Adam(faster_rcnn.parameters(), lr = 0.0001)

for epoch in range(100):
    for image, data in tests:
        for key in data:
            data[key] = data[key].cuda()

        optimizer.zero_grad()
        loss = faster_rcnn(image.unsqueeze(0).cuda(), [data])
        losses = 0
        for key in loss:
            losses += loss[key]
        losses.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(epoch, losses)

faster_rcnn.eval()
out = faster_rcnn(image.unsqueeze(0).cuda())
out = out[0]
torch.save(faster_rcnn.state_dict(), '/src/faster_rcnn.mdl')

src.utils.render_boxes(image.unsqueeze(0), out, 0.5)
