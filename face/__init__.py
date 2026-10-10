"""
face package: Chứa các thành phần tiền xử lý và trích xuất đặc trưng khuôn mặt
"""
from .preprocess import FacePreprocess
from .feature_extractor import FaceFeatureExtractor

__all__ = ["FacePreprocess", "FaceFeatureExtractor"]
