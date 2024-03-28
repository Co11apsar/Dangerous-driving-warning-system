import torch

class CigaRecog:
    def __init__(self):
        self.model = torch.hub.load('C:/study/yolov5/yolov5', 'custom', './weights/ciga.pt', source='local')
        # self.model.conf = 0.3

    def predict(self, img):
        results = self.model(img[:, :, ::-1])
        pd = results.pandas().xyxy[0]
        ciga_pd = pd[pd['class'] == 0]
        box_list = ciga_pd.to_numpy()

        predict_res = []
        for box in box_list:
            l, t = int(box[0]), int(box[1])
            r, b = int(box[2]), int(box[3])
            conf = box[4]

            predict_res.append(((l, t, r, b), conf))

        return predict_res
