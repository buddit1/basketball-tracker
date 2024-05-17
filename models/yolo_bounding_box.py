from ultralytics import YOLO
import torch
import numpy as np





class YOLOBasketballBB():
    def __init__(self, base_model, device='cpu'):
        self.model = YOLO(base_model + '.pt')
        self.model.fuse()
        self.device = device

    def __call__(self, frame, *args, **kwargs):
        preds = self.model.predict(frame, device=self.device, classes=[32], verbose=False, half=True)
        ball_idxs = torch.argwhere(preds[0].boxes.cls == 32)
        ball_xyxy = preds[0].boxes.xyxy[ball_idxs].cpu() if len(ball_idxs) != 0 else None
        return ball_xyxy
        
