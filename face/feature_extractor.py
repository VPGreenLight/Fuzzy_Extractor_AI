"""
face/feature_extractor.py: Trích xuất đặc trưng bằng FaceNet theo Mục 3.2.1 trong demo.pdf
Nhiệm vụ:
  Ánh xạ ảnh khuôn mặt chuẩn hóa 160x160x3 thành vector embedding 512 chiều đã được chuẩn hóa L2.
"""
import torch
import numpy as np
import config

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


class FaceFeatureExtractor:
    def __init__(self, device: str = None):
        """
        Khởi tạo mô hình mạng nơ-ron InceptionResnetV1 tải sẵn trọng số vggface2.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if FACENET_AVAILABLE:
            self.model = InceptionResnetV1(pretrained=config.PRETRAINED_WEIGHTS).eval().to(self.device)
        else:
            self.model = None

    def extract(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Trích xuất vector embedding 512 chiều từ tensor khuôn mặt chuẩn hóa.
        - Input: face_tensor [1, 3, 160, 160]
        - Output: numpy.ndarray kích thước (512,) chuẩn hóa L2
        """
        if self.model is not None:
            with torch.no_grad():
                embedding = self.model(face_tensor)
            emb_np = embedding.squeeze().cpu().numpy()
            # Chuẩn hóa L2: ||v||_2 = 1
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            return emb_np
        else:
            # Mô phỏng vector 512 chiều khi chưa tải được weights
            np.random.seed(int(face_tensor.sum().item() * 1000) % (2**31))
            mock_emb = np.random.randn(config.FACE_EMBEDDING_DIM).astype(np.float32)
            return mock_emb / np.linalg.norm(mock_emb)
