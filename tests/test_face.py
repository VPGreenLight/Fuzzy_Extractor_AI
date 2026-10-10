"""
tests/test_face.py: Kiểm thử mở rộng toàn diện cho Module 1: face/
"""
import unittest
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from face.preprocess import FacePreprocess
from face.feature_extractor import FaceFeatureExtractor
import config


class TestFaceModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = FacePreprocess()
        cls.extractor = FaceFeatureExtractor()

        # Đường dẫn ảnh thực tế trong data/face/
        cls.face_dir = config.FACE_DATA_DIR
        cls.img_a1 = cls.face_dir / "personA_1.jpg"
        cls.img_a2 = cls.face_dir / "personA_2.jpg"
        cls.img_b1 = cls.face_dir / "personB_1.jpg"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.blank_img_path = os.path.join(self.temp_dir, "blank_no_face.jpg")
        dummy_img = Image.new("RGB", (200, 200), color=(200, 180, 160))
        dummy_img.save(self.blank_img_path)

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_feature_extraction_dimension_and_norm(self):
        """Kiểm tra InceptionResnetV1 trích xuất vector 512 chiều chuẩn hóa L2"""
        dummy_tensor = torch.randn(1, 3, config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE)
        embedding = self.extractor.extract(dummy_tensor)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (config.FACE_EMBEDDING_DIM,))
        l2_norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_02_batch_extraction(self):
        """Kiểm tra trích xuất đặc trưng hàng loạt theo lô (Batch)"""
        batch_size = 4
        dummy_batch = torch.randn(batch_size, 3, config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE)
        embeddings = self.extractor.extract_batch(dummy_batch)

        self.assertEqual(embeddings.shape, (batch_size, config.FACE_EMBEDDING_DIM))
        for row in embeddings:
            self.assertAlmostEqual(np.linalg.norm(row), 1.0, places=4)

    def test_03_preprocess_rejects_non_face(self):
        """Kiểm tra MTCNN phát hiện và từ chối ảnh không có khuôn mặt"""
        with self.assertRaises(ValueError):
            self.preprocessor.process(self.blank_img_path)

    def test_04_preprocess_crops_valid_face(self):
        """Kiểm tra MTCNN cắt và căn chỉnh ảnh khuôn mặt thực tế đạt chuẩn 160x160"""
        if not self.img_a1.exists():
            self.skipTest(f"Không tìm thấy ảnh mẫu {self.img_a1}")

        crop_save_path = os.path.join(self.temp_dir, "test_crop.jpg")
        tensor = self.preprocessor.process(self.img_a1, save_crop_path=crop_save_path)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE))
        self.assertTrue(os.path.exists(crop_save_path), "Ảnh crop debug phải được lưu thành công!")

    def test_05_distance_metrics_math(self):
        """Kiểm tra tính toán số học của khoảng cách Euclidean và Cosine"""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Trùng nhau
        self.assertEqual(self.extractor.euclidean_distance(vec1, vec1), 0.0)
        self.assertAlmostEqual(self.extractor.cosine_similarity(vec1, vec1), 1.0, places=4)

        # Trực giao
        self.assertAlmostEqual(self.extractor.cosine_similarity(vec1, vec2), 0.0, places=4)
        self.assertAlmostEqual(self.extractor.euclidean_distance(vec1, vec2), np.sqrt(2), places=4)

    def test_06_biometric_discrimination(self):
        """Kiểm tra độ phân biệt sinh trắc học: Cùng người (d < 0.6) vs Khác người (d > 1.0)"""
        if not (self.img_a1.exists() and self.img_a2.exists() and self.img_b1.exists()):
            self.skipTest("Cần đủ 3 ảnh mẫu để thực hiện kiểm thử độ phân biệt sinh trắc học")

        tensor_a1 = self.preprocessor.process(self.img_a1)
        tensor_a2 = self.preprocessor.process(self.img_a2)
        tensor_b1 = self.preprocessor.process(self.img_b1)

        emb_a1 = self.extractor.extract(tensor_a1)
        emb_a2 = self.extractor.extract(tensor_a2)
        emb_b1 = self.extractor.extract(tensor_b1)

        d_same = self.extractor.euclidean_distance(emb_a1, emb_a2)
        d_diff = self.extractor.euclidean_distance(emb_a1, emb_b1)

        # Kiểm chứng khoảng cách lý thuyết
        self.assertLess(d_same, 0.6, f"Khoảng cách cùng 1 người ({d_same:.4f}) phải nhỏ hơn ngưỡng 0.6!")
        self.assertGreater(d_diff, 1.0, f"Khoảng cách 2 người khác nhau ({d_diff:.4f}) phải lớn hơn 1.0!")

        # Kiểm chứng phán đoán nhị phân
        self.assertTrue(self.extractor.is_same_person(emb_a1, emb_a2))
        self.assertFalse(self.extractor.is_same_person(emb_a1, emb_b1))


if __name__ == "__main__":
    unittest.main()
