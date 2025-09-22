# video_pipeline.py
# video_pipeline.py
import av
import torch
import numpy as np
from torchvision.transforms import Resize, Normalize

class VideoMAE_ONNX_Pipeline:
    def __init__(self, onnx_path, num_frames=16, resize_to=(224,224), mean=None, std=None, id2label=None):
        import onnxruntime as ort
        if id2label is None:
            raise ValueError("id2label은 반드시 전달해야 합니다.")
        self.ort_sess = ort.InferenceSession(onnx_path)
        self.num_frames = num_frames
        self.resize = Resize(resize_to)
        self.normalize = Normalize(mean=mean or [0.485,0.456,0.406], std=std or [0.229,0.224,0.225])
        self.id2label = id2label

    def __call__(self, video_path):
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames or 300  # 프레임 정보 없으면 임시 300

        # 샘플링 프레임 인덱스
        indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_image()  # PIL Image
                img_tensor = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
                img_tensor = self.normalize(self.resize(img_tensor))
                frames.append(img_tensor)
            if len(frames) == self.num_frames:
                break

        if len(frames) < self.num_frames:
            # 부족하면 마지막 프레임 복제
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        video_tensor = torch.stack(frames).unsqueeze(0)
        ort_inputs = {"pixel_values": video_tensor.numpy().astype(np.float32)}
        logits = self.ort_sess.run(None, ort_inputs)[0]
        pred_id = int(np.argmax(logits, axis=1)[0])
        return self.id2label[pred_id]

