"""
face/preprocess.py: Tiền xử lý ảnh khuôn mặt theo Mục 3.2.1 trong demo.pdf
Nhiệm vụ:
  1. Đọc và kiểm tra tính hợp lệ của ảnh đầu vào.
  2. Định vị, cắt (crop) và căn chỉnh (align) vùng khuôn mặt bằng MTCNN.
  3. Chuẩn hóa kích thước về 160x160x3 và chuyển thành Tensor.
"""
from PIL import Image
import torch
import numpy as np
import config

try:
    from facenet_pytorch import MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


class FacePreprocess:
    def __init__(self, device: str = None):
        """
        Khởi tạo pipeline tiền xử lý ảnh khuôn mặt.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if FACENET_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=config.FACE_IMAGE_SIZE,
                margin=14,
                keep_all=False,
                device=self.device
            )
        else:
            self.mtcnn = None

    def process(self, image_path: str) -> torch.Tensor:
        """
        Tiền xử lý ảnh khuôn mặt:
        - Input: image_path (đường dẫn tới file ảnh)
        - Output: torch.Tensor kích thước [1, 3, 160, 160]
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}. Chi tiết lỗi: {e}")

        if self.mtcnn is not None:
            face_tensor = self.mtcnn(img)
            if face_tensor is None:
                raise ValueError(f"Không phát hiện được khuôn mặt hợp lệ trong ảnh: {image_path}")
            return face_tensor.unsqueeze(0).to(self.device)
        else:
            # Fallback căn bản khi facenet-pytorch chưa được cài đặt
            img_resized = img.resize((config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # Chuẩn hóa về [-1, 1]
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return tensor
