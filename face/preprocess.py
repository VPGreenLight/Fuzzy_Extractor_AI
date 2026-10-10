"""
face/preprocess.py: Tiền xử lý ảnh khuôn mặt theo Mục 3.2.1 trong demo.pdf
Nhiệm vụ:
  1. Đọc và chuẩn hóa đa dạng định dạng đầu vào (Path, str, PIL.Image, numpy.ndarray).
  2. Định vị, cắt (crop) và căn chỉnh (align) khuôn mặt bằng MTCNN.
  3. Xử lý đa khuôn mặt: Tự động chọn khuôn mặt có diện tích lớn nhất (chủ thể ở cận cảnh).
  4. Chuẩn hóa kích thước về 160x160x3 và chuyển đổi sang Tensor [-1, 1].
  5. Cung cấp hàm tiện ích lưu ảnh crop để phục vụ báo cáo / slide.
"""
from pathlib import Path
from typing import Union, Tuple, Optional
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
    def __init__(self, device: str = None, margin: int = 20):
        """
        Khởi tạo pipeline tiền xử lý ảnh khuôn mặt.
        - margin: Số pixel đệm thêm xung quanh khuôn mặt (mặc định 20 để giữ trọn cằm/trán)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.margin = margin

        if FACENET_AVAILABLE:
            # select_largest=True: tự động chọn khuôn mặt lớn nhất khi ảnh có nhiều người
            self.mtcnn = MTCNN(
                image_size=config.FACE_IMAGE_SIZE,
                margin=self.margin,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                select_largest=True,
                post_process=True,
                device=self.device
            )
        else:
            self.mtcnn = None

    def _load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Chuẩn hóa mọi kiểu đầu vào về đối tượng PIL.Image (RGB)
        """
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"Không tìm thấy file ảnh tại đường dẫn: {path_obj}")
            try:
                return Image.open(path_obj).convert('RGB')
            except Exception as e:
                raise ValueError(f"Không thể đọc file ảnh '{path_obj}': {e}")
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Nếu là ma trận ảnh OpenCV (BGR hoặc RGB)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Giả định mặc định OpenCV BGR -> chuyển sang RGB
                return Image.fromarray(image_input[:, :, ::-1])
            elif len(image_input.shape) == 2:
                return Image.fromarray(image_input).convert('RGB')
            else:
                return Image.fromarray(image_input)
        else:
            raise TypeError(f"Kiểu dữ liệu ảnh không hỗ trợ: {type(image_input)}")

    def process(self, image_input: Union[str, Path, Image.Image, np.ndarray], 
                save_crop_path: Optional[Union[str, Path]] = None) -> torch.Tensor:
        """
        Tiền xử lý ảnh khuôn mặt:
        - Input:
            + image_input: đường dẫn file, PIL Image hoặc numpy array
            + save_crop_path: đường dẫn lưu ảnh crop 160x160 (nếu muốn debug/lưu báo cáo)
        - Output: torch.Tensor kích thước [1, 3, 160, 160] trong dải [-1, 1]
        """
        img = self._load_image(image_input)

        if self.mtcnn is not None:
            # Chuẩn bị đường dẫn lưu ảnh crop nếu có yêu cầu
            save_str = None
            if save_crop_path is not None:
                p = Path(save_crop_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                save_str = str(p)

            # MTCNN crop & align
            face_tensor = self.mtcnn(img, save_path=save_str)

            if face_tensor is None:
                source_name = str(image_input) if isinstance(image_input, (str, Path)) else "ảnh đầu vào"
                raise ValueError(f"Không phát hiện được khuôn mặt hợp lệ trong {source_name}!")

            return face_tensor.unsqueeze(0).to(self.device)
        else:
            # Fallback khi chưa có facenet-pytorch
            img_resized = img.resize((config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE))
            if save_crop_path:
                img_resized.save(save_crop_path)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return tensor

    def detect_landmarks(self, image_input: Union[str, Path, Image.Image, np.ndarray]):
        """
        Trả về bounding box và 5 điểm mốc khuôn mặt (2 mắt, mũi, 2 khóe miệng)
        """
        if self.mtcnn is None:
            raise RuntimeError("MTCNN không khả dụng để phát hiện điểm mốc.")
        img = self._load_image(image_input)
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        return boxes, probs, landmarks

    def save_debug_crop(self, image_input: Union[str, Path, Image.Image, np.ndarray], 
                        output_path: Union[str, Path]) -> Path:
        """
        Tiện ích cắt và lưu khuôn mặt căn chỉnh ra đĩa phục vụ báo cáo / slide
        """
        out_p = Path(output_path)
        self.process(image_input, save_crop_path=out_p)
        return out_p
