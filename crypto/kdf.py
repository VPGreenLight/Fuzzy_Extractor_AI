"""
crypto/kdf.py: Dẫn xuất khóa HKDF-HMAC-SHA256 theo Mục 2.3.3 và 3.2.4 demo.pdf
Nhiệm vụ:
  Chuyển đổi chuỗi bí mật R thành khóa AES-256 có độ dài chuẩn 32 bytes (256 bits).
"""
import hmac
import hashlib
import math
import config

try:
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class KeyDerivation:
    @staticmethod
    def derive_key(secret_R: bytes, salt: bytes = config.KDF_SALT, 
                   info: bytes = config.KDF_INFO, length: int = config.AES_KEY_BYTES) -> bytes:
        """
        Dẫn xuất khóa qua HKDF (HMAC-SHA256) gồm 2 giai đoạn: Extract và Expand.
        - Input:
            + secret_R: Chuỗi bí mật sinh ra từ Fuzzy Extractor
            + salt: Giá trị muối phụ trợ
            + info: Chuỗi ngữ cảnh phân định mục đích sử dụng khóa
            + length: Chiều dài khóa mong muốn (mặc định 32 bytes cho AES-256)
        - Output: bytes khóa AES-256 độ dài 32 bytes
        """
        if HAS_CRYPTOGRAPHY:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                info=info
            )
            return hkdf.derive(secret_R)
        else:
            # Triển khai chuẩn RFC 5869 bằng hmac + hashlib của thư viện chuẩn Python
            # 1. HKDF-Extract: PRK = HMAC-Hash(salt, IKM)
            if not salt:
                salt = bytes([0] * 32)
            prk = hmac.new(salt, secret_R, hashlib.sha256).digest()

            # 2. HKDF-Expand: T(1) = HMAC-Hash(PRK, info || 0x01)...
            n = math.ceil(length / 32)
            t = b""
            okm = b""
            for i in range(1, n + 1):
                t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
                okm += t

            return okm[:length]
