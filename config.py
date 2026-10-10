"""
config.py: Lưu các tham số cấu hình dùng chung toàn hệ thống
Căn cứ theo Mục 3.6.2 trong báo cáo demo.pdf
"""
from pathlib import Path

# Đường dẫn thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Các thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
FACE_DATA_DIR = DATA_DIR / "face"
FINGERPRINT_DATA_DIR = DATA_DIR / "fingerprint"
ENCRYPTED_DATA_DIR = DATA_DIR / "encrypted"

# Thư mục lưu trữ lâu dài (Helper Data và Metadata)
STORAGE_DIR = BASE_DIR / "storage_data"

# Tự động tạo thư mục nếu chưa có
for dir_path in [DATA_DIR, FACE_DATA_DIR, FINGERPRINT_DATA_DIR, ENCRYPTED_DATA_DIR, STORAGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. CẤU HÌNH TIỀN XỬ LÝ & MÔ HÌNH HỌC SÂU (FACE / FINGERPRINT)
# ==============================================================================
FACE_IMAGE_SIZE = 160            # Kích thước ảnh chuẩn sau căn chỉnh MTCNN (160x160x3)
FACE_EMBEDDING_DIM = 512         # Số chiều vector đặc trưng từ FaceNet (InceptionResnetV1)
PRETRAINED_WEIGHTS = 'vggface2'  # Trọng số pre-trained huấn luyện trên VGGFace2

FINGERPRINT_FEATURE_DIM = 512   # Số chiều vector đặc trưng vân tay

# ==============================================================================
# 2. CẤU HÌNH LƯỢNG TỬ HÓA & MÃ SỬA LỖI (FUZZY EXTRACTOR)
# ==============================================================================
QUANTIZED_BITS = 128             # Số bit nhị phân sau lượng tử hóa (16 bytes)
BCH_M = 8                        # Trường Galois GF(2^8)
BCH_ERROR_CORRECTION_BITS = 12   # Khả năng sửa lỗi tối đa t = 12 bits (~10% theo demo.pdf)
BCH_DATA_BYTES = 4               # Độ dài bí mật R: 4 bytes (32 bits)
BCH_ECC_BYTES = 12               # Độ dài ECC: 12 bytes (96 bits) -> Tổng: 16 bytes = 128 bits

# ==============================================================================
# 3. CẤU HÌNH MẬT MÃ HỌC (HKDF & AES-256-GCM)
# ==============================================================================
KDF_SALT = b"PTIT_INT1344_BIOMETRIC_SALT_2026"
KDF_INFO = b"AES_256_GCM_ENCRYPTION_KEY"
AES_KEY_BYTES = 32               # Độ dài khóa AES-256: 32 bytes (256 bits)
GCM_IV_BYTES = 12                # Độ dài giá trị khởi tạo IV chuẩn GCM: 12 bytes (96 bits)
GCM_TAG_BYTES = 16               # Độ dài thẻ xác thực Tag: 16 bytes (128 bits)
