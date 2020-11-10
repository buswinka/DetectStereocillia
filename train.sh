#!/bin/bash
python main.py --epochs $1 --train_mask_rcnn --train_data ./data/train --lr 5e-6 --pretrained_model ./models/mask_rcnn_REALLYGOOD.mdl
