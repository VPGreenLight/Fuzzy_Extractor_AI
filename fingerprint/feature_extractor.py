"""
fingerprint/feature_extractor.py: Trích xuất đặc trưng vân tay theo Mục 3.2.2 trong demo.pdf
Nhiệm vụ:
  Chuyển đổi ảnh vân tay đã tiền xử lý thành vector đặc trưng cố định 512 chiều.
"""
import cv2
import numpy as np
import config


class FingerprintFeatureExtractor:
    @staticmethod
    def extract(processed_img: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector đặc trưng vân tay 512 chiều:
        - Input: processed_img (ảnh vân tay đã xử lý, ma trận 256x256)
        - Output: numpy.ndarray kích thước (512,) chuẩn hóa Z-score
        """
        # Giảm số chiều không gian ma trận ảnh về kích thước 32x16 = 512 chiều
        downsampled = cv2.resize(processed_img, (32, 16), interpolation=cv2.INTER_AREA)
        features = downsampled.flatten().astype(np.float32)

        # Chuẩn hóa Z-score để đưa về phân bố xấp xỉ N(0, 1)
        mean_val = np.mean(features)
        std_val = np.std(features) + 1e-7
        norm_features = (features - mean_val) / std_val

        # Cắt lấy đúng số chiều cấu hình
        return norm_features[:config.FINGERPRINT_FEATURE_DIM]
