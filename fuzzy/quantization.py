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

    def select_reliable_indices(self, embedding: np.ndarray) -> np.ndarray:
        """
        Chọn lọc các chiều đặc trưng ổn định nhất (Reliable Bit Selection - Mục 2.1.5 demo.pdf):
        Lựa chọn target_bits thành phần có biên độ tuyệt đối |f_i| lớn nhất (xa ngưỡng 0.0 nhất)
        để giảm thiểu hiện tượng lật bit do nhiễu ánh sáng, góc chụp hoặc biểu cảm.
        """
        if len(embedding) < self.target_bits:
            raise ValueError(f"Vector đặc trưng có độ dài ({len(embedding)}) nhỏ hơn target_bits ({self.target_bits})!")
        # Sắp xếp tăng dần theo |f_i|, lấy target_bits vị trí lớn nhất và giữ thứ tự chỉ số tăng dần
        top_indices = np.argsort(np.abs(embedding))[-self.target_bits:]
        return np.sort(top_indices).astype(np.int32)

    def quantize(self, embedding: np.ndarray, threshold: float = 0.0,
                 reliable_indices: np.ndarray = None) -> np.ndarray:
        """
        Lượng tử hóa 1-bit dựa trên ngưỡng (Thresholding):
          b_i = 1 nếu f_i >= threshold, ngược lại b_i = 0
        - Input:
            + embedding: vector số thực chiều >= target_bits
            + threshold: ngưỡng phân cực (mặc định 0.0)
            + reliable_indices: danh sách chỉ số các chiều ổn định (nếu có từ Helper Data)
        - Output: numpy.ndarray chứa các giá trị uint8 {0, 1} độ dài target_bits
        """
        if len(embedding) < self.target_bits:
            raise ValueError(f"Vector đặc trưng có độ dài ({len(embedding)}) nhỏ hơn target_bits ({self.target_bits})!")

        if reliable_indices is not None:
            selected = embedding[np.asarray(reliable_indices, dtype=np.int32)]
        else:
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
