"""
main.py: Tệp điều phối toàn bộ chương trình theo Mục 3.6.2 demo.pdf
Cung cấp giao diện dòng lệnh (CLI) thực hiện:
  1. Đăng ký & Mã hóa tệp tin (--mode enroll)
  2. Tái tạo khóa & Giải mã tệp tin (--mode decrypt)
  3. Chạy toàn bộ các bài kiểm thử đơn vị (--mode test)
"""
import sys
# Hỗ trợ hiển thị tiếng Việt trên Windows terminal mà không bị lỗi UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import unittest
from pathlib import Path

from face.preprocess import FacePreprocess
from face.feature_extractor import FaceFeatureExtractor
from fingerprint.preprocess import FingerprintPreprocess
from fingerprint.feature_extractor import FingerprintFeatureExtractor
from fuzzy.quantization import Quantizer
from fuzzy.fuzzy_extractor import FuzzyExtractor
from crypto.kdf import KeyDerivation
from crypto.aes_gcm import AESGCMCipher
from storage.helper_data import HelperDataStorage
from storage.metadata import MetadataStorage
import config


def get_biometric_bits(image_path: str, modality: str = "face",
                       reliable_indices=None) -> tuple:
    """
    Quy trình Tiền xử lý -> Trích xuất AI -> Lượng tử hóa thành chuỗi bit
    - Nếu reliable_indices=None (giai đoạn đăng ký): Tự động chọn lọc 128 chiều ổn định nhất.
    - Nếu có reliable_indices (giai đoạn tái tạo): Lượng tử hóa theo đúng các chiều trong Helper Data.
    """
    quantizer = Quantizer(target_bits=config.QUANTIZED_BITS)

    if modality == "face":
        print(f"  [1.1] Tiền xử lý ảnh khuôn mặt bằng MTCNN: {image_path}...")
        preprocessor = FacePreprocess()
        tensor = preprocessor.process(image_path)

        print(f"  [1.2] Trích xuất vector 512 chiều bằng FaceNet (InceptionResnetV1)...")
        extractor = FaceFeatureExtractor()
        embedding = extractor.extract(tensor)
    elif modality == "fingerprint":
        print(f"  [1.1] Tiền xử lý ảnh vân tay (CLAHE, Gaussian): {image_path}...")
        processed = FingerprintPreprocess.process(image_path)

        print(f"  [1.2] Trích xuất biểu diễn đặc trưng vân tay...")
        embedding = FingerprintFeatureExtractor.extract(processed)
    else:
        raise ValueError(f"Loại sinh trắc học không hỗ trợ: {modality}")

    if reliable_indices is None:
        reliable_indices = quantizer.select_reliable_indices(embedding)

    print(f"  [1.3] Lượng tử hóa vector thành chuỗi bit nhị phân {config.QUANTIZED_BITS}-bit...")
    w = quantizer.quantize(embedding, reliable_indices=reliable_indices)
    return w, embedding, reliable_indices


def enroll(profile_id: str, image_path: str, file_to_encrypt: str, modality: str = "face"):
    """
    Quy trình Đăng ký sinh trắc học và Mã hóa tệp tin (Mục 3.3 demo.pdf)
    """
    print(f"\n==================================================================")
    print(f"  BẮT ĐẦU QUY TRÌNH ĐĂNG KÝ & MÃ HÓA (Hồ sơ: {profile_id} - Nhánh: {modality})")
    print(f"==================================================================")

    # 1. Thu thập và lượng tử hóa đặc trưng sinh trắc học
    print("\n[Bước 1] Thu thập và lượng tử hóa sinh trắc học:")
    w, _, reliable_indices = get_biometric_bits(image_path, modality)

    # 2. Sinh bí mật R và Helper Data P qua Fuzzy Extractor Gen
    print("\n[Bước 2] Thực thi thuật toán Fuzzy Extractor Gen(w):")
    fuzzy = FuzzyExtractor()
    secret_R, helper_data = fuzzy.gen(w, reliable_indices=reliable_indices)
    print(f"  -> Sinh bí mật ngẫu nhiên R: {len(secret_R)} bytes")
    print(f"  -> Tính Helper Data P (bảo vệ bằng mã sửa lỗi BCH, t={config.BCH_ERROR_CORRECTION_BITS})")

    # 3. Dẫn xuất khóa AES-256 qua HKDF-HMAC-SHA256
    print("\n[Bước 3] Dẫn xuất khóa AES-256 bằng HKDF (RFC 5869):")
    aes_key = KeyDerivation.derive_key(secret_R, config.KDF_SALT, config.KDF_INFO)
    print(f"  -> Tạo thành công khóa đối xứng AES-256 (32 bytes = 256 bits)")

    # 4. Mã hóa tệp tin bằng AES-256-GCM
    output_enc = config.ENCRYPTED_DATA_DIR / f"{profile_id}_{modality}_data.enc"
    print(f"\n[Bước 4] Mã hóa tệp dữ liệu '{file_to_encrypt}' với AES-256-GCM:")
    iv, tag = AESGCMCipher.encrypt_file(file_to_encrypt, str(output_enc), aes_key)
    print(f"  -> Sinh giá trị khởi tạo IV ngẫu nhiên: {len(iv)} bytes (96 bits)")
    print(f"  -> Sinh thẻ xác thực Tag: {len(tag)} bytes (128 bits)")
    print(f"  -> Bản mã được lưu tại: {output_enc}")

    # 5. Lưu trữ Helper Data P và Metadata
    print("\n[Bước 5] Lưu trữ dữ liệu phụ trợ lâu dài:")
    p_path = HelperDataStorage.save(profile_id, modality, helper_data)
    m_path = MetadataStorage.save(profile_id, modality, iv, tag, config.KDF_SALT, config.KDF_INFO)
    print(f"  -> Helper Data P: {p_path.name}")
    print(f"  -> Metadata: {m_path.name}")

    print("\n------------------------------------------------------------------")
    print("=> KẾT QUẢ: ĐĂNG KÝ VÀ MÃ HÓA THÀNH CÔNG!")
    print("   (Lưu ý: Bí mật R và khóa AES đã được hủy tức thời khỏi bộ nhớ RAM)")
    print("------------------------------------------------------------------\n")


def decrypt(profile_id: str, new_image_path: str, output_decrypted_file: str, modality: str = "face"):
    """
    Quy trình Tái tạo khóa và Giải mã tệp tin (Mục 3.4 demo.pdf)
    """
    print(f"\n==================================================================")
    print(f"  BẮT ĐẦU QUY TRÌNH TÁI TẠO KHÓA & GIẢI MÃ (Hồ sơ: {profile_id} - Nhánh: {modality})")
    print(f"==================================================================")

    # 1. Tải Helper Data P trước để lấy cấu hình các chiều lượng tử hóa ổn định (nếu có)
    helper_data = HelperDataStorage.load(profile_id, modality)
    reliable_indices = helper_data.get("reliable_indices", None)

    # 2. Thu thập mẫu sinh trắc học mới
    print("\n[Bước 1] Thu thập mẫu sinh trắc học xác thực mới:")
    w_prime, _, _ = get_biometric_bits(new_image_path, modality, reliable_indices=reliable_indices)

    print("\n[Bước 2] Tải Helper Data P từ storage:")
    print(f"  -> Tải thành công Helper Data cho hồ sơ '{profile_id}'")

    # 3. Tái tạo bí mật R bằng Fuzzy Extractor Rep (sửa lỗi BCH)
    print(f"\n[Bước 3] Thực thi Fuzzy Extractor Rep(w', P) với mã sửa sai BCH:")
    fuzzy = FuzzyExtractor()
    recovered_R = fuzzy.rep(w_prime, helper_data)
    print(f"  -> Sửa sai thành công! Khôi phục chính xác chuỗi bí mật ban đầu R")

    # 4. Tái tạo khóa AES-256 qua HKDF
    print("\n[Bước 4] Tái tạo khóa AES-256 từ bí mật R khôi phục qua HKDF:")
    meta = MetadataStorage.load(profile_id, modality)
    recovered_key = KeyDerivation.derive_key(recovered_R, meta["salt"], meta["info"])
    print(f"  -> Tái tạo thành công khóa đối xứng AES-256")

    # 5. Giải mã và xác thực thẻ Tag AES-GCM
    enc_file = config.ENCRYPTED_DATA_DIR / f"{profile_id}_{modality}_data.enc"
    print(f"\n[Bước 5] Giải mã tệp tin '{enc_file.name}' và xác thực tính toàn vẹn:")
    AESGCMCipher.decrypt_file(str(enc_file), output_decrypted_file, recovered_key,
                              meta["iv"], meta["tag"], meta["aad"])

    print("\n------------------------------------------------------------------")
    print(f"=> KẾT QUẢ: GIẢI MÃ THÀNH CÔNG VÀ TOÀN VẸN!")
    print(f"   Tệp tin đã được khôi phục tại: {output_decrypted_file}")
    print("------------------------------------------------------------------\n")


def run_unit_tests():
    """Chạy toàn bộ các bài kiểm thử độc lập trong tests/"""
    print("\n=== ĐANG CHẠY BỘ KIỂM THỬ ĐỘC LẬP (UNIT TESTS) ===")
    loader = unittest.TestLoader()
    suite = loader.discover(str(config.BASE_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống Sinh khóa Mật mã Sinh trắc học kết hợp AI & Fuzzy Extractor - PTIT INT1344"
    )
    parser.add_argument("--mode", choices=["enroll", "decrypt", "test"], required=True,
                        help="Chế độ thực thi: enroll (đăng ký), decrypt (giải mã), test (kiểm thử)")
    parser.add_argument("--modality", choices=["face", "fingerprint"], default="face",
                        help="Loại sinh trắc học: face (khuôn mặt) hoặc fingerprint (vân tay)")
    parser.add_argument("--profile", default="user01",
                        help="Định danh hồ sơ người dùng (ví dụ: user01)")
    parser.add_argument("--image",
                        help="Đường dẫn file ảnh sinh trắc học")
    parser.add_argument("--input-file",
                        help="Đường dẫn file gốc cần mã hóa (dành cho mode enroll)")
    parser.add_argument("--output-file",
                        help="Đường dẫn file sau khi giải mã (dành cho mode decrypt)")

    args = parser.parse_args()

    if args.mode == "test":
        run_unit_tests()

    if not args.image:
        print("Lỗi: Chế độ 'enroll' và 'decrypt' yêu cầu cung cấp tham số --image <đường dẫn ảnh>")
        sys.exit(1)

    try:
        if args.mode == "enroll":
            if not args.input_file:
                print("Lỗi: Chế độ 'enroll' yêu cầu cung cấp tham số --input-file <file cần mã hóa>")
                sys.exit(1)
            enroll(args.profile, args.image, args.input_file, args.modality)
        elif args.mode == "decrypt":
            if not args.output_file:
                print("Lỗi: Chế độ 'decrypt' yêu cầu cung cấp tham số --output-file <file đầu ra>")
                sys.exit(1)
            decrypt(args.profile, args.image, args.output_file, args.modality)
    except Exception as e:
        print(f"\n[!] THẤT BẠI: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
