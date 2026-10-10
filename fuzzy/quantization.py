"""
fuzzy/quantization.py: Chuẩn hóa và lượng tử hóa đặc trưng theo Mục 2.1.5 và 3.2.3 demo.pdf
Nhiệm vụ:
  1. Chuyển đổi vector số thực liên tục f thành chuỗi bit nhị phân rời rạc w in {0, 1}^n.
  2. Cung cấp hàm đo khoảng cách Hamming d_H(w, w') để đánh giá mức độ sai lệch bit.
"""
import numpy as np
import config


class Quantizer:
    def __init__(self, target_bits: int = config.QUANTIZED_BITS):
        """
        Khởi tạo bộ lượng tử hóa với số lượng bit mục tiêu.
        """
        self.target_bits = target_bits

    def quantize(self, embedding: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Lượng tử hóa 1-bit dựa trên ngưỡng (Thresholding):
          b_i = 1 nếu f_i >= threshold, ngược lại b_i = 0
        - Input: embedding (vector số thực chiều >= target_bits)
        - Output: numpy.ndarray chứa các giá trị uint8 {0, 1} độ dài target_bits
        """
        if len(embedding) < self.target_bits:
            raise ValueError(f"Vector đặc trưng có độ dài ({len(embedding)}) nhỏ hơn target_bits ({self.target_bits})!")

        selected = embedding[:self.target_bits]
        bits = (selected >= threshold).astype(np.uint8)
        return bits

    @staticmethod
    def hamming_distance(w1: np.ndarray, w2: np.ndarray) -> int:
        """
        Tính khoảng cách Hamming d_H(w1, w2) - số vị trí bit khác nhau giữa hai chuỗi bit.
        """
        if len(w1) != len(w2):
            raise ValueError("Hai chuỗi bit phải có cùng độ dài để tính khoảng cách Hamming!")
        return int(np.sum(w1 != w2))
