import torchvision.models
"""
list[Dict[Tensor]]<- mask_rcnn(boxes=FloatTensor[N,4], labels=Int64Tensor[N], masks=UInt8Tensor[N,H,W])

IN:
boxes: [x1,y1,x2,y2]

        boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
        labels (Int64Tensor[N]): the class label for each ground-truth box
        masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

OUT:
boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
    with values of x between 0 and W and values of y between 0 and H
labels (Int64Tensor[N]): the predicted labels for each mask
scores (Tensor[N]): the scores or each prediction
masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, 
    in 0-1 range. In order to obtain the final segmentation masks, 
    the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)

"""
# num_classes always has to be 1 greater than what you think
mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=4,
                                                               pretrained_backbone=True)

faster_rcnn_bundle = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2,
                                                               pretrained_backbone=True)

faster_rcnn_cilia = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=8,
                                                               pretrained_backbone=True,
                                                                         trainable_backbone_layers=5)

keypoint_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2,
                                                                       num_keypoints=3, pretrained_backbone=True,
                                                                       trainable_backbone_layers=3)

