"""
fuzzy/fuzzy_extractor.py: Cài đặt thuật toán Fuzzy Extractor theo Mục 2.2.2 và 3.2.3 trong demo.pdf
Nhiệm vụ:
  1. Hàm Gen(w) -> (R, P): Sinh chuỗi bí mật R và dữ liệu phụ trợ Helper Data P từ chuỗi bit w.
  2. Hàm Rep(w', P) -> R: Tái tạo lại chính xác chuỗi bí mật R từ chuỗi bit mới w' và Helper Data P.
"""
import os
import numpy as np
from .ecc import ErrorCorrectingCode
import config


class FuzzyExtractor:
    def __init__(self):
        """
        Khởi tạo hệ thống Fuzzy Extractor với cấu hình tham số từ config.py.
        """
        self.ecc = ErrorCorrectingCode()
        self.bits_len = config.QUANTIZED_BITS

    def gen(self, w: np.ndarray) -> tuple[bytes, dict]:
        """
        Hàm khởi tạo (Generate - Gen):
        - Input: Chuỗi bit sinh trắc học w độ dài config.QUANTIZED_BITS
        - Output:
            + secret_R: Chuỗi bí mật ngẫu nhiên R (bytes)
            + helper_data: Dữ liệu phụ trợ Helper Data P (dict) để lưu trữ công khai
        """
        if len(w) != self.bits_len:
            raise ValueError(f"Độ dài chuỗi bit w ({len(w)}) không khớp cấu hình ({self.bits_len})!")

        # 1. Sinh ngẫu nhiên chuỗi bí mật R (độ dài data_len của bộ mã sửa sai)
        secret_R = os.urandom(self.ecc.data_len)

        # 2. Tạo từ mã bảo vệ: Codeword C = [R || ECC(R)]
        ecc_bytes = self.ecc.encode(secret_R)
        codeword_bytes = secret_R + ecc_bytes

        # Chuyển từ mã sang chuỗi bit
        codeword_bits = np.unpackbits(np.frombuffer(codeword_bytes, dtype=np.uint8))[:self.bits_len]

        # 3. Tính Helper Data: P = w XOR C
        helper_bits = np.bitwise_xor(w, codeword_bits)

        helper_data = {
            "helper_bits": helper_bits.tolist(),
            "bits_len": self.bits_len,
            "data_len": self.ecc.data_len,
            "ecc_len": self.ecc.ecc_len
        }

        return secret_R, helper_data

    def rep(self, w_prime: np.ndarray, helper_data: dict) -> bytes:
        """
        Hàm tái tạo (Reproduce - Rep):
        - Input:
            + w_prime: Chuỗi bit mới từ lần quét sau (có nhiễu)
            + helper_data: Dữ liệu phụ trợ P đã lưu từ giai đoạn Gen
        - Output: Chuỗi bí mật R được khôi phục chính xác (nếu sai lệch <= t)
        """
        if len(w_prime) != self.bits_len:
            raise ValueError(f"Độ dài chuỗi bit w_prime ({len(w_prime)}) không khớp ({self.bits_len})!")

        helper_bits = np.array(helper_data["helper_bits"], dtype=np.uint8)

        # 1. Tính từ mã có lỗi: C' = P XOR w' = C XOR (w XOR w')
        noisy_codeword_bits = np.bitwise_xor(helper_bits, w_prime)

        # Chuyển bit về mảng byte
        pad_len = (8 - (len(noisy_codeword_bits) % 8)) % 8
        if pad_len > 0:
            padded_bits = np.pad(noisy_codeword_bits, (0, pad_len), 'constant')
        else:
            padded_bits = noisy_codeword_bits

        codeword_bytes = np.packbits(padded_bits).tobytes()

        data_len = helper_data.get("data_len", self.ecc.data_len)
        ecc_len = helper_data.get("ecc_len", self.ecc.ecc_len)

        recv_data = bytearray(codeword_bytes[:data_len])
        recv_ecc = bytearray(codeword_bytes[data_len:data_len + ecc_len])

        # 2. Sử dụng thuật toán giải mã lỗi BCH để khôi phục R
        bitflips, fixed_R = self.ecc.decode_and_correct(recv_data, recv_ecc)

        if bitflips < 0:
            raise ValueError(
                f"Tái tạo thất bại! Mức sai lệch bit vượt quá khả năng sửa sai "
                f"(ngưỡng t = {config.BCH_ERROR_CORRECTION_BITS} bits)."
            )

        return fixed_R
