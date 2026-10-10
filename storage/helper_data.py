"""
storage/helper_data.py: Quản lý lưu trữ và truy xuất Helper Data P theo Mục 3.5.2 demo.pdf
Nhiệm vụ:
  1. Lưu trữ Helper Data P ra file JSON an toàn theo từng định danh hồ sơ profile_id.
  2. Truy xuất Helper Data P để phục vụ quy trình tái tạo khóa (Rep).
"""
import json
from pathlib import Path
import config


class HelperDataStorage:
    @staticmethod
    def get_filepath(profile_id: str, modality: str) -> Path:
        return config.STORAGE_DIR / f"{profile_id}_{modality}_helper.json"

    @classmethod
    def save(cls, profile_id: str, modality: str, helper_data: dict) -> Path:
        """
        Lưu Helper Data P ra file JSON.
        """
        filepath = cls.get_filepath(profile_id, modality)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(helper_data, f, indent=4)
        return filepath

    @classmethod
    def load(cls, profile_id: str, modality: str) -> dict:
        """
        Tải Helper Data P từ file JSON.
        """
        filepath = cls.get_filepath(profile_id, modality)
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy Helper Data cho hồ sơ '{profile_id}' (nhánh {modality}) tại: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
