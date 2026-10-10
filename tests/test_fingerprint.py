"""
tests/test_fingerprint.py: Kiểm thử độc lập nhánh sinh trắc vân tay
"""
import unittest
import numpy as np
import cv2
import tempfile
import os
from fingerprint.preprocess import FingerprintPreprocess
from fingerprint.feature_extractor import FingerprintFeatureExtractor
import config


class TestFingerprintModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.temp_dir, "synthetic_fingerprint.png")
        # Tạo ảnh vân tay nhân tạo (sọc vân trắng đen)
        arr = np.zeros((300, 300), dtype=np.uint8)
        for i in range(0, 300, 10):
            cv2.line(arr, (0, i), (300, i), color=255, thickness=2)
        cv2.imwrite(self.test_img_path, arr)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
        os.rmdir(self.temp_dir)

    def test_fingerprint_pipeline(self):
        """Kiểm tra quy trình tiền xử lý và trích xuất vector đặc trưng vân tay"""
        processed = FingerprintPreprocess.process(self.test_img_path)
        self.assertEqual(processed.shape, (256, 256))

        features = FingerprintFeatureExtractor.extract(processed)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), config.FINGERPRINT_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
