from ultralytics import YOLO
import numpy as np





class YOLOBasketballBB():
    def __init__(self, base_model, device='cpu'):
        self.model = YOLO(base_model + '.pt')
        self.model.fuse()
        self.device = device

    def find_balls(self, frame, **kwargs):
        """
        Detect any sports balls in image. Arbitrary kwargs can be passed to the underlying yolo model's
        predict function. See (yolo inference arguments)[https://docs.ultralytics.com/modes/predict/#inference-arguments]

        If any balls are detected then function returns
        a tensor of shape n x 3 where n is the number of detected balls in the image.
        each entry in the tensor is of the format [x, y, radius] describing the position and size of
        a ball in the image.
        If no balls detected then function returns None.
        """
        preds = self.model.predict(frame, device=self.device, classes=[32], verbose=False, half=True, **kwargs)
        ball_xyxys = preds[0].boxes.xyxy.cpu().numpy()
        balls = np.empty((ball_xyxys.shape[0], 3), dtype=np.int32) if len(ball_xyxys) else None
        if len(ball_xyxys) > 0:
            balls[:,0] = (ball_xyxys[:,0] + ball_xyxys[:,2]) // 2
            balls[:,1] =  (ball_xyxys[:,1] + ball_xyxys[:,3]) // 2
            #calculate radius as half average of height and width
            balls[:,2] = ((ball_xyxys[:,2] - ball_xyxys[:,0] ) + (ball_xyxys[:,3] - ball_xyxys[:,1])) // 4
        return balls
        
    def predict(self, frame, **kwargs):
        return self.model.predict(frame, device=self.device, classes=[32], verbose=False, half=True, **kwargs)
