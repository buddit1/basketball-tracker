from ultralytics import YOLO
import numpy as np





class YOLOBasketballBB():
    def __init__(self, base_model, device='cpu'):
        self.model = YOLO(base_model + '.pt')
        self.model.fuse()
        self.device = device

    def find_ball_centers(self, frame):
        preds = self.model.predict(frame, device=self.device, classes=[32], verbose=False, half=True)
        ball_xyxys = preds[0].boxes.xyxy.cpu().numpy()
        centers = np.empty((ball_xyxys.shape[0], 2), dtype=np.int32) if len(ball_xyxys) else None
        if len(ball_xyxys) > 0:
            centers[:,0] = (ball_xyxys[:,0] + ball_xyxys[:,2]) // 2
            centers[:,1] =  (ball_xyxys[:,1] + ball_xyxys[:,3]) // 2
        return centers
        
    def predict(self, frame):
        return self.model.predict(frame, device=self.device, classes=[32], verbose=False, half=True)
