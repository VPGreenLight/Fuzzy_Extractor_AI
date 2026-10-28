"""
fingerprint/preprocess.py: Tiền xử lý ảnh vân tay theo Mục 3.2.2 trong demo.pdf
Nhiệm vụ:
  1. Đọc đa định dạng ảnh vân tay (.tif, .png, .jpg, Path, PIL.Image, numpy.ndarray) an toàn Unicode.
  2. Tách nền (Foreground Segmentation) và căn tâm vùng lõi vân tay (ROI Center-of-Mass Alignment)
     để triệt tiêu độ lệch vị trí đặt ngón tay giữa các lần quét trên cảm biến (như FVC2000 DB4_B).
  3. Chuẩn hóa kích thước về 256x256, nâng cao tương phản cục bộ bằng CLAHE và lọc nhiễu Gaussian.
"""
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import cv2
import numpy as np


class FingerprintPreprocess:
    @staticmethod
    def _load_grayscale(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Đọc ảnh vân tay về mảng numpy grayscale uint8, hỗ trợ đường dẫn tiếng Việt Unicode và file .tif (FVC2000).
        """
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise ValueError(f"Không tìm thấy ảnh vân tay tại: {path_obj}")
            try:
                pil_img = Image.open(path_obj).convert("L")
                return np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Không thể đọc ảnh vân tay tại '{path_obj}': {e}")
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("L"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            return image_input.astype(np.uint8)
        else:
            raise TypeError(f"Kiểu dữ liệu ảnh vân tay không hỗ trợ: {type(image_input)}")

    @staticmethod
    def _align_roi(img: np.ndarray) -> np.ndarray:
        """
        Tách nền vùng chứa đường vân và căn tâm theo trọng tâm cường độ vân (Weighted Center-of-Mass).
        Giúp ổn định vùng đặc trưng khi ngón tay đặt lệch vị trí trên mặt cảm biến FVC2000.
        """
        inv = 255 - img
        blur_inv = cv2.GaussianBlur(inv, (15, 15), 0)
        _, mask = cv2.threshold(blur_inv, 25, 255, cv2.THRESH_BINARY)

        # Loại bỏ nhiễu sát mép viền cảm biến
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        ys, xs = np.where(mask > 0)
        if len(ys) > 100:
            weights = blur_inv[ys, xs].astype(np.float64) + 1.0
            cy = int(np.sum(ys * weights) / np.sum(weights))
            cx = int(np.sum(xs * weights) / np.sum(weights))

            half_h = min(max(int(img.shape[0] * 0.35), 80), 120)
            half_w = min(max(int(img.shape[1] * 0.40), 80), 110)

            pad = 200
            padded = np.pad(img, pad, mode="constant", constant_values=255)
            cy_p, cx_p = cy + pad, cx + pad
            crop = padded[cy_p - half_h : cy_p + half_h, cx_p - half_w : cx_p + half_w]
            if crop.size > 0:
                return crop
        return img

    @classmethod
    def process(cls, image_input: Union[str, Path, Image.Image, np.ndarray],
                save_crop_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Tiền xử lý ảnh vân tay:
        - Input:
            + image_input: đường dẫn file (.tif, .png, .jpg) hoặc mảng ảnh
            + save_crop_path: đường dẫn lưu ảnh vân tay đã căn tâm & tăng cường (phục vụ báo cáo)
        - Output: numpy.ndarray (256, 256) đã được căn tâm ROI, tăng tương phản CLAHE và lọc nhiễu
        """
        img = cls._load_grayscale(image_input)

        # 1. Căn tâm vùng lõi vân tay (ROI Alignment)
        roi = cls._align_roi(img)

        # 2. Chuẩn hóa kích thước ảnh về chuẩn 256x256
        resized = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)

        # 3. Cân bằng biểu đồ độ sáng cục bộ thích ứng (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # 4. Lọc nhiễu Gaussian làm mượt đường vân (ridge/valley)
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

        if save_crop_path is not None:
            out_p = Path(save_crop_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(denoised).save(out_p)

        return denoised
