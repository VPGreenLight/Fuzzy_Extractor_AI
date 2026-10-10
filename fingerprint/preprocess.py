"""
fingerprint/preprocess.py: Tiền xử lý ảnh vân tay theo Mục 3.2.2 trong demo.pdf
Nhiệm vụ:
  1. Đọc ảnh vân tay thang độ xám (grayscale).
  2. Nâng cao độ tương phản cục bộ bằng CLAHE.
  3. Lọc nhiễu Gaussian để làm rõ các đường vân (ridge/valley).
"""
import cv2
import numpy as np


class FingerprintPreprocess:
    @staticmethod
    def process(image_path: str) -> np.ndarray:
        """
        Tiền xử lý ảnh vân tay:
        - Input: image_path (đường dẫn tới file ảnh vân tay)
        - Output: numpy.ndarray ảnh đã được lọc nhiễu và tăng tương phản
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh vân tay tại: {image_path}")

        # Chuẩn hóa kích thước ảnh về chuẩn 256x256
        resized = cv2.resize(img, (256, 256))

        # Cân bằng biểu đồ độ sáng cục bộ thích ứng (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # Lọc nhiễu Gaussian
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        return denoised
