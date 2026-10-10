"""
tests/test_fuzzy.py: Kiểm thử độc lập module lượng tử hóa và thuật toán Fuzzy Extractor (BCH)
"""
import unittest
import numpy as np
from fuzzy.quantization import Quantizer
from fuzzy.fuzzy_extractor import FuzzyExtractor
import config


class TestFuzzyModule(unittest.TestCase):
    def setUp(self):
        self.quantizer = Quantizer(target_bits=config.QUANTIZED_BITS)
        self.fuzzy = FuzzyExtractor()

    def test_quantization_and_hamming(self):
        """Kiểm tra lượng tử hóa vector float sang bit và tính khoảng cách Hamming"""
        vec1 = np.ones(config.QUANTIZED_BITS, dtype=np.float32)
        vec2 = -np.ones(config.QUANTIZED_BITS, dtype=np.float32)

        bits1 = self.quantizer.quantize(vec1)
        bits2 = self.quantizer.quantize(vec2)

        self.assertTrue(np.all(bits1 == 1))
        self.assertTrue(np.all(bits2 == 0))
        self.assertEqual(self.quantizer.hamming_distance(bits1, bits2), config.QUANTIZED_BITS)

    def test_fuzzy_extractor_reconstruction_success(self):
        """Kiểm tra tái tạo thành công khi sai khác bit nằm trong ngưỡng t (<= 10 bits)"""
        w = np.random.randint(0, 2, size=config.QUANTIZED_BITS, dtype=np.uint8)
        secret_R, helper_data = self.fuzzy.gen(w)

        # Mô phỏng nhiễu: Lật 6 bit ngẫu nhiên
        w_noisy = w.copy()
        error_indices = [2, 10, 25, 50, 75, 110]
        w_noisy[error_indices] ^= 1

        hamming_dist = self.quantizer.hamming_distance(w, w_noisy)
        self.assertEqual(hamming_dist, len(error_indices))

        # Tái tạo bí mật
        recovered_R = self.fuzzy.rep(w_noisy, helper_data)
        self.assertEqual(secret_R, recovered_R, "Bí mật khôi phục phải khớp 100% với bí mật ban đầu!")

    def test_fuzzy_extractor_failure_exceed_threshold(self):
        """Kiểm tra hệ thống báo lỗi khi sai khác vượt quá khả năng sửa sai (> t bits)"""
        w = np.random.randint(0, 2, size=config.QUANTIZED_BITS, dtype=np.uint8)
        _, helper_data = self.fuzzy.gen(w)

        # Lật 50 bit (sai khác ngẫu nhiên như giữa 2 người khác nhau)
        w_diff_person = w.copy()
        flip_indices = np.random.choice(config.QUANTIZED_BITS, size=50, replace=False)
        w_diff_person[flip_indices] ^= 1

        with self.assertRaises(ValueError):
            self.fuzzy.rep(w_diff_person, helper_data)


if __name__ == "__main__":
    unittest.main()
