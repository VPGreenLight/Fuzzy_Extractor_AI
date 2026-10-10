"""
crypto/aes_gcm.py: Mã hóa và giải mã AES-256-GCM theo Mục 2.3.2 và 3.2.4 demo.pdf
Nhiệm vụ:
  1. Mã hóa đối xứng có xác thực (AEAD) với khóa AES 256-bit (32 bytes).
  2. Tạo giá trị khởi tạo IV 96-bit (12 bytes) ngẫu nhiên, không tái sử dụng.
  3. Tạo thẻ xác thực Tag 128-bit (16 bytes) và kiểm tra tính toàn vẹn khi giải mã.
"""
import os
import config

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AESGCM = True
except ImportError:
    HAS_AESGCM = False


class AESGCMCipher:
    @staticmethod
    def encrypt_file(file_path: str, output_path: str, key: bytes, 
                     aad: bytes = None) -> tuple[bytes, bytes]:
        """
        Mã hóa tệp bằng AES-256-GCM:
        - Input:
            + file_path: Đường dẫn tệp tin gốc cần mã hóa
            + output_path: Đường dẫn tệp tin sau mã hóa (.enc)
            + key: Khóa AES-256 (32 bytes)
            + aad: Dữ liệu liên kết cần xác thực (nếu có)
        - Output: tuple (iv, tag)
            + iv: Giá trị khởi tạo 12 bytes (96 bits)
            + tag: Thẻ xác thực 16 bytes (128 bits)
        """
        if not HAS_AESGCM:
            raise ImportError("Thư viện 'cryptography' chưa được cài đặt. Vui lòng chạy: pip install cryptography")

        if len(key) != config.AES_KEY_BYTES:
            raise ValueError(f"Khóa AES phải có độ dài đúng {config.AES_KEY_BYTES} bytes ({config.AES_KEY_BYTES * 8} bits)!")

        aesgcm = AESGCM(key)
        # Sinh IV ngẫu nhiên 96-bit (12 bytes) cho mỗi lần mã hóa (Mục 2.3.2)
        iv = os.urandom(config.GCM_IV_BYTES)

        with open(file_path, 'rb') as f:
            plaintext = f.read()

        # Trong cryptography Python, hàm encrypt trả về: Ciphertext || Tag (16 bytes cuối)
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, aad)
        ciphertext = ciphertext_with_tag[:-config.GCM_TAG_BYTES]
        tag = ciphertext_with_tag[-config.GCM_TAG_BYTES:]

        with open(output_path, 'wb') as f:
            f.write(ciphertext)

        return iv, tag

    @staticmethod
    def decrypt_file(encrypted_path: str, output_path: str, key: bytes, 
                     iv: bytes, tag: bytes, aad: bytes = None):
        """
        Giải mã tệp và kiểm tra tính toàn vẹn qua thẻ xác thực Tag:
        - Input:
            + encrypted_path: Đường dẫn tệp tin mã hóa (.enc)
            + output_path: Đường dẫn tệp tin sau giải mã
            + key: Khóa AES-256 (32 bytes)
            + iv: Giá trị khởi tạo 12 bytes
            + tag: Thẻ xác thực 16 bytes
            + aad: Dữ liệu liên kết xác thực
        - Output: None (ghi tệp giải mã ra output_path nếu thành công)
        - Exception: Ném ValueError nếu Tag không hợp lệ hoặc dữ liệu bị sửa đổi
        """
        if not HAS_AESGCM:
            raise ImportError("Thư viện 'cryptography' chưa được cài đặt. Vui lòng chạy: pip install cryptography")

        if len(key) != config.AES_KEY_BYTES:
            raise ValueError(f"Khóa AES phải có độ dài đúng {config.AES_KEY_BYTES} bytes!")

        with open(encrypted_path, 'rb') as f:
            ciphertext = f.read()

        aesgcm = AESGCM(key)
        ciphertext_with_tag = ciphertext + tag

        try:
            plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, aad)
        except Exception:
            raise ValueError(
                "Giải mã thất bại! Khóa mật mã không đúng hoặc bản mã/thẻ Tag đã bị thay đổi trái phép (Tag Verification Failed)."
            )

        with open(output_path, 'wb') as f:
            f.write(plaintext)
