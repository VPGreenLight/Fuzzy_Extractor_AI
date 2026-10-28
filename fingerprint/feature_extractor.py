"""
fingerprint/feature_extractor.py: Trích xuất đặc trưng vân tay theo Mục 3.2.2 trong demo.pdf
Nhiệm vụ:
  Sử dụng ngân hàng bộ lọc Gabor cầu phương 8 hướng (8-Orientation Quadrature Gabor Filterbank - FingerCode)
  để trích xuất vector đặc trưng cấu trúc đường vân 512 chiều (8 hướng x lưới không gian 8x8),
  bất biến với pha đường vân/rãnh vân và chuẩn hóa L2.
"""
import cv2
import numpy as np
import config


class FingerprintFeatureExtractor:
    # Khởi tạo sẵn ngân hàng 8 cặp bộ lọc Gabor cầu phương (Even/Odd) giúp tăng tốc độ trích xuất
    _GABOR_BANK = None
    _HANNING_WIN = None

    @classmethod
    def _init_kernels(cls):
        if cls._GABOR_BANK is not None:
            return
        cls._HANNING_WIN = np.outer(np.hanning(256), np.hanning(256)).astype(np.float32)
        bank = []
        for k in range(8):
            theta = k * np.pi / 8.0
            kern_re = cv2.getGaborKernel(
                (25, 25), sigma=5.0, theta=theta, lambd=10.0, gamma=0.5, psi=0.0, ktype=cv2.CV_32F
            )
            kern_im = cv2.getGaborKernel(
                (25, 25), sigma=5.0, theta=theta, lambd=10.0, gamma=0.5, psi=np.pi / 2.0, ktype=cv2.CV_32F
            )
            # Khử thành phần DC (zero-mean) để bộ lọc chỉ phản hồi với cấu trúc đường vân
            kern_re -= kern_re.mean()
            kern_im -= kern_im.mean()
            bank.append((kern_re, kern_im))
        cls._GABOR_BANK = bank

    @classmethod
    def extract(cls, processed_img: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector đặc trưng vân tay 512 chiều (Gabor FingerCode):
        - Input: processed_img (ảnh vân tay đã tiền xử lý, ma trận 256x256)
        - Output: numpy.ndarray kích thước (512,) chuẩn hóa L2 (||f||_2 = 1.0)
        """
        cls._init_kernels()

        if processed_img.shape != (256, 256):
            processed_img = cv2.resize(processed_img, (256, 256), interpolation=cv2.INTER_AREA)

        # Áp cửa sổ Hanning triệt tiêu hiệu ứng biên
        img_zero = (processed_img.astype(np.float32) - float(np.mean(processed_img))) * cls._HANNING_WIN

        energies = []
        for kern_re, kern_im in cls._GABOR_BANK:
            r_re = cv2.filter2D(img_zero, cv2.CV_32F, kern_re)
            r_im = cv2.filter2D(img_zero, cv2.CV_32F, kern_im)
            # Năng lượng bao hình cầu phương (bất biến pha giữa đỉnh vân và rãnh vân)
            mag = np.sqrt(r_re * r_re + r_im * r_im)
            mag_smooth = cv2.GaussianBlur(mag, (35, 35), 9.0)
            # Lấy vùng lõi trung tâm [32:224, 32:224] chia thành lưới 8x8 ô
            grid_mag = cv2.resize(mag_smooth[32:224, 32:224], (8, 8), interpolation=cv2.INTER_AREA)
            energies.append(grid_mag)

        # Tensor đặc trưng (8 hướng, 8 hàng, 8 cột) = 512 chiều
        E = np.stack(energies, axis=0)

        # Chuẩn hóa zero-mean theo hướng tại từng ô lưới và theo không gian
        E = E - np.mean(E, axis=0, keepdims=True)
        spatial_std = np.std(E, axis=(1, 2), keepdims=True)
        if np.all(spatial_std > 1e-3):
            E = E - np.mean(E, axis=(1, 2), keepdims=True)

        features = E.flatten().astype(np.float32)
        l2_norm = float(np.linalg.norm(features))
        if l2_norm > 1e-7:
            features = features / l2_norm

        return features[:config.FINGERPRINT_FEATURE_DIM]

    @staticmethod
    def euclidean_distance(f1: np.ndarray, f2: np.ndarray) -> float:
        return float(np.linalg.norm(f1 - f2))

    @staticmethod
    def cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2)) + 1e-9
        return float(np.dot(f1, f2) / denom)
