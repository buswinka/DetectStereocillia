from src.model import mask_rcnn
from src.dataloader import MaskData
import src.transforms as t
import src.utils
import torch.optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

transforms = torchvision.transforms.Compose([
    t.random_h_flip(),
    t.random_v_flip(),
    t.random_affine(),
    t.adjust_brightness(),
    t.adjust_contrast(),
    t.correct_boxes()
])

tests = MaskData('../data/train/', transforms=transforms)
tests = DataLoader(tests, batch_size=None, shuffle=True, num_workers=4)

mask_rcnn = mask_rcnn.train().cuda()
# mask_rcnn.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/tests/mask_rcnn.mdl'))
optimizer = torch.optim.Adam(mask_rcnn.parameters(), lr = 5e-5)

all_loss = []

for epoch in range(1001):
    epoch_loss = []
    for image, data in tests:
        for key in data:
            data[key] = data[key].cuda()

        optimizer.zero_grad()
        loss = mask_rcnn(image.unsqueeze(0).cuda(), [data])
        losses = 0
        for key in loss:
            losses += loss[key]
        losses.backward()
        epoch_loss.append(losses.item())
        all_loss.append(losses.item())
        optimizer.step()

    if epoch % 25 == 0:
        print(f'Epoch:{epoch} | loss: {torch.tensor(epoch_loss).sum().item()}')

plt.plot(all_loss)
plt.show()

mask_rcnn.eval()
out = mask_rcnn(image.unsqueeze(0).cuda())
out = out[0]

src.utils.render_mask(image.unsqueeze(0), out, 0.5)
src.utils.render_mask(image.unsqueeze(0), data,0.5)
torch.save(mask_rcnn.state_dict(), '/src/mask_rcnn.mdl')