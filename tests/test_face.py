"""
tests/test_face.py: Kiểm thử độc lập nhánh sinh trắc khuôn mặt (MTCNN & FaceNet)
"""
import unittest
import numpy as np
from PIL import Image
import tempfile
import os
import torch
from face.preprocess import FacePreprocess
from face.feature_extractor import FaceFeatureExtractor
import config


class TestFaceModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.blank_img_path = os.path.join(self.temp_dir, "blank_no_face.jpg")
        # Tạo ảnh trơn không có khuôn mặt
        dummy_img = Image.new("RGB", (200, 200), color=(200, 180, 160))
        dummy_img.save(self.blank_img_path)

        self.preprocessor = FacePreprocess()
        self.extractor = FaceFeatureExtractor()

    def tearDown(self):
        if os.path.exists(self.blank_img_path):
            os.remove(self.blank_img_path)
        os.rmdir(self.temp_dir)

    def test_preprocess_rejects_non_face(self):
        """Kiểm tra MTCNN từ chối ảnh không có khuôn mặt hợp lệ"""
        with self.assertRaises(ValueError):
            self.preprocessor.process(self.blank_img_path)

    def test_feature_extraction_dimension_and_norm(self):
        """Kiểm tra InceptionResnetV1 trích xuất vector 512 chiều chuẩn hóa L2"""
        # Tạo tensor chuẩn 160x160x3
        dummy_face_tensor = torch.randn(1, 3, config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE)
        embedding = self.extractor.extract(dummy_face_tensor)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), config.FACE_EMBEDDING_DIM)

        # Kiểm tra chuẩn hóa L2: ||embedding||_2 xấp xỉ 1.0
        l2_norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(l2_norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
