"""
tests/test_crypto.py: Kiểm thử độc lập module mật mã (HKDF và AES-256-GCM)
"""
import unittest
import os
import tempfile
from crypto.kdf import KeyDerivation
from crypto.aes_gcm import AESGCMCipher
import config


class TestCryptoModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.plain_file = os.path.join(self.temp_dir, "plain.txt")
        self.enc_file = os.path.join(self.temp_dir, "cipher.enc")
        self.dec_file = os.path.join(self.temp_dir, "decrypted.txt")

        self.original_content = b"Hoc phan Mat ma hoc co so - PTIT 2026. Bao ve toan ven du lieu!"
        with open(self.plain_file, "wb") as f:
            f.write(self.original_content)

    def tearDown(self):
        for p in [self.plain_file, self.enc_file, self.dec_file]:
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(self.temp_dir)

    def test_hkdf_determinism_and_length(self):
        """Kiểm tra HKDF sinh đúng 32 bytes (256 bits) và tính tất định"""
        secret = os.urandom(config.BCH_DATA_BYTES)
        key1 = KeyDerivation.derive_key(secret, config.KDF_SALT, config.KDF_INFO)
        key2 = KeyDerivation.derive_key(secret, config.KDF_SALT, config.KDF_INFO)

        self.assertEqual(len(key1), config.AES_KEY_BYTES)
        self.assertEqual(key1, key2, "Cùng đầu vào bí mật và tham số phải sinh ra cùng một khóa AES!")

        # Thay đổi salt thì khóa phải khác
        key_different_salt = KeyDerivation.derive_key(secret, salt=b"DIFFERENT_SALT")
        self.assertNotEqual(key1, key_different_salt)

    def test_aes_gcm_encrypt_decrypt_integrity(self):
        """Kiểm tra mã hóa và giải mã tệp bằng AES-256-GCM bảo toàn nội dung"""
        secret = os.urandom(config.BCH_DATA_BYTES)
        key = KeyDerivation.derive_key(secret)

        iv, tag = AESGCMCipher.encrypt_file(self.plain_file, self.enc_file, key)
        self.assertEqual(len(iv), config.GCM_IV_BYTES)
        self.assertEqual(len(tag), config.GCM_TAG_BYTES)

        AESGCMCipher.decrypt_file(self.enc_file, self.dec_file, key, iv, tag)
        with open(self.dec_file, "rb") as f:
            recovered_content = f.read()

        self.assertEqual(self.original_content, recovered_content)

    def test_aes_gcm_tamper_detection(self):
        """Kiểm tra cơ chế phát hiện sửa đổi bản mã hoặc thẻ Tag (Authentication Tag check)"""
        secret = os.urandom(config.BCH_DATA_BYTES)
        key = KeyDerivation.derive_key(secret)

        iv, tag = AESGCMCipher.encrypt_file(self.plain_file, self.enc_file, key)

        # Cố ý làm sai lệch 1 byte trong thẻ Tag
        tampered_tag = bytearray(tag)
        tampered_tag[0] ^= 0xFF

        with self.assertRaises(ValueError):
            AESGCMCipher.decrypt_file(self.enc_file, self.dec_file, key, iv, bytes(tampered_tag))


if __name__ == "__main__":
    unittest.main()
