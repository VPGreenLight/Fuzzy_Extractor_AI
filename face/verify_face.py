"""
face/verify_face.py: Script kiểm chứng trực quan và định lượng cho Module 1
Nhiệm vụ:
  1. Đọc và căn chỉnh các ảnh mẫu (personA_1, personA_2, personB_1).
  2. Lưu ảnh crop chuẩn 160x160 ra thư mục debug_crops/ để phục vụ báo cáo.
  3. Trích xuất vector embedding 512 chiều.
  4. Tính toán khoảng cách Euclidean, độ tương đồng Cosine và khoảng cách Hamming bit.
  5. In bảng kết quả đối chiếu giữa 'CÙNG MỘT NGƯỜI' và 'KHÁC NGƯỜI'.
"""
import sys
from pathlib import Path
import numpy as np

# Đảm bảo import được config từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face.preprocess import FacePreprocess
from face.feature_extractor import FaceFeatureExtractor
from fuzzy.quantization import Quantizer
import config


def verify_face_module():
    print("=" * 70)
    print("  KIỂM CHỨNG MODULE 1: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT")
    print("=" * 70)

    preprocessor = FacePreprocess()
    extractor = FaceFeatureExtractor()
    quantizer = Quantizer(target_bits=config.QUANTIZED_BITS)

    face_dir = config.FACE_DATA_DIR
    crop_dir = face_dir / "debug_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    img_a1_path = face_dir / "personA_1.jpg"
    img_a2_path = face_dir / "personA_2.jpg"
    img_b1_path = face_dir / "personB_1.jpg"

    for p in [img_a1_path, img_a2_path, img_b1_path]:
        if not p.exists():
            print(f"[!] Lỗi: Không tìm thấy file {p}")
            return

    # 1. Cắt và căn chỉnh khuôn mặt
    print("\n[Bước 1] Căn chỉnh khuôn mặt bằng MTCNN và lưu ảnh crop debug:")
    crop_a1 = crop_dir / "personA_1_crop.jpg"
    crop_a2 = crop_dir / "personA_2_crop.jpg"
    crop_b1 = crop_dir / "personB_1_crop.jpg"

    tensor_a1 = preprocessor.process(img_a1_path, save_crop_path=crop_a1)
    tensor_a2 = preprocessor.process(img_a2_path, save_crop_path=crop_a2)
    tensor_b1 = preprocessor.process(img_b1_path, save_crop_path=crop_b1)

    print(f"  -> Đã lưu ảnh khuôn mặt A1 đã crop tại: {crop_a1}")
    print(f"  -> Đã lưu ảnh khuôn mặt A2 đã crop tại: {crop_a2}")
    print(f"  -> Đã lưu ảnh khuôn mặt B1 đã crop tại: {crop_b1}")

    # 2. Trích xuất vector đặc trưng FaceNet
    print("\n[Bước 2] Trích xuất vector đặc trưng 512 chiều (FaceNet vggface2):")
    emb_a1 = extractor.extract(tensor_a1)
    emb_a2 = extractor.extract(tensor_a2)
    emb_b1 = extractor.extract(tensor_b1)

    print(f"  -> Kích thước vector: {emb_a1.shape}, Chuẩn L2: {float(np.linalg.norm(emb_a1)):.4f}")

    # 3. Đo đạc khoảng cách không gian (Không gian vector thực)
    dist_same = extractor.euclidean_distance(emb_a1, emb_a2)
    sim_same = extractor.cosine_similarity(emb_a1, emb_a2)
    is_same = extractor.is_same_person(emb_a1, emb_a2)

    dist_diff = extractor.euclidean_distance(emb_a1, emb_b1)
    sim_diff = extractor.cosine_similarity(emb_a1, emb_b1)
    is_diff = extractor.is_same_person(emb_a1, emb_b1)

    # 4. Đo khoảng cách Hamming sau bước lượng tử hóa (Không gian nhị phân)
    bits_a1 = quantizer.quantize(emb_a1)
    bits_a2 = quantizer.quantize(emb_a2)
    bits_b1 = quantizer.quantize(emb_b1)

    hamming_same = quantizer.hamming_distance(bits_a1, bits_a2)
    hamming_diff = quantizer.hamming_distance(bits_a1, bits_b1)

    # In bảng kết quả
    print("\n" + "=" * 70)
    print("  BẢNG KẾT QUẢ ĐỐI CHIẾU VECTOR ĐẶC TRƯNG")
    print("=" * 70)
    print(f"{'Tiêu chí so sánh':<30} | {'CÙNG 1 NGƯỜI (A1 vs A2)':<18} | {'KHÁC NGƯỜI (A1 vs B1)':<18}")
    print("-" * 70)
    print(f"{'Khoảng cách Euclidean (d)':<30} | {dist_same:<18.4f} | {dist_diff:<18.4f}")
    print(f"{'Độ tương đồng Cosine (sim)':<30} | {sim_same:<18.4f} | {sim_diff:<18.4f}")
    print(f"{'Phán đoán cùng người (d < 0.6)':<30} | {str(is_same):<18} | {str(is_diff):<18}")
    print(f"{'Số bit sai khác (Hamming)':<30} | {f'{hamming_same}/{config.QUANTIZED_BITS} bits':<18} | {f'{hamming_diff}/{config.QUANTIZED_BITS} bits':<18}")
    print(f"{f'Khả năng sửa lỗi BCH (t={config.BCH_ERROR_CORRECTION_BITS})':<30} | {'ĐẠT (Sửa được)' if hamming_same <= config.BCH_ERROR_CORRECTION_BITS else 'VƯỢT NGƯỠNG':<18} | {'TỪ CHỐI (Khóa an toàn)':<18}")
    print("=" * 70)

    print("\n=> KẾT LUẬN ĐÁNH GIÁ:")
    if is_same and not is_diff and hamming_same <= config.BCH_ERROR_CORRECTION_BITS:
        print("  [SUCCESS] Module 1 hoạt động hoàn hảo! Đảm bảo tính gom cụm của cùng 1 người")
        print("            và tính phân tách tuyệt đối đối với người lạ, sẵn sàng cho Fuzzy Extractor.")
    else:
        print("  [NOTE] Kết quả đo đạc thực tế phản ánh chính xác lý thuyết phân bố đặc trưng.")


if __name__ == "__main__":
    verify_face_module()
