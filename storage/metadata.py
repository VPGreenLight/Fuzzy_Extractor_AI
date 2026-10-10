"""
storage/metadata.py: Quản lý siêu dữ liệu mật mã theo Mục 3.5.2 demo.pdf
Nhiệm vụ:
  Quản lý các tham số kỹ thuật phục vụ phiên giải mã:
  - Giá trị khởi tạo IV (96-bit)
  - Thẻ xác thực Tag (128-bit)
  - Tham số KDF (salt, info)
  - Dữ liệu liên kết AAD (nếu có)
"""
import json
import base64
from pathlib import Path
import config


class MetadataStorage:
    @staticmethod
    def get_filepath(profile_id: str, modality: str) -> Path:
        return config.STORAGE_DIR / f"{profile_id}_{modality}_meta.json"

    @classmethod
    def save(cls, profile_id: str, modality: str, iv: bytes, tag: bytes, 
             salt: bytes, info: bytes, aad: bytes = None) -> Path:
        """
        Lưu metadata ra file JSON (mã hóa base64 các trường bytes).
        """
        metadata = {
            "profile_id": profile_id,
            "modality": modality,
            "iv": base64.b64encode(iv).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8'),
            "salt": base64.b64encode(salt).decode('utf-8'),
            "info": base64.b64encode(info).decode('utf-8'),
            "aad": base64.b64encode(aad).decode('utf-8') if aad else None
        }
        filepath = cls.get_filepath(profile_id, modality)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        return filepath

    @classmethod
    def load(cls, profile_id: str, modality: str) -> dict:
        """
        Tải và giải mã base64 các tham số metadata từ file JSON.
        """
        filepath = cls.get_filepath(profile_id, modality)
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy Metadata cho hồ sơ '{profile_id}' (nhánh {modality}) tại: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        return {
            "profile_id": raw["profile_id"],
            "modality": raw["modality"],
            "iv": base64.b64decode(raw["iv"]),
            "tag": base64.b64decode(raw["tag"]),
            "salt": base64.b64decode(raw["salt"]),
            "info": base64.b64decode(raw["info"]),
            "aad": base64.b64decode(raw["aad"]) if raw.get("aad") else None
        }
