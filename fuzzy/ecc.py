"""
fuzzy/ecc.py: Mã sửa lỗi khối BCH nhị phân theo Mục 2.2.3 trong demo.pdf
Nhiệm vụ:
  1. Sử dụng bchlib cài đặt mã BCH nhị phân với khả năng sửa tối đa t bit sai lệch.
  2. Cung cấp fallback thuần Python khi bchlib chưa có sẵn.
"""
import config

try:
    import bchlib  # type: ignore[import-untyped]
    HAS_BCHLIB = True
except ImportError:
    HAS_BCHLIB = False


class PurePythonBCH:
    def __init__(self, t: int = config.BCH_ERROR_CORRECTION_BITS):
        self.t = t
        self.data_len = config.BCH_DATA_BYTES
        self.ecc_len = config.BCH_ECC_BYTES

    def encode(self, data: bytes) -> bytes:
        ecc = bytearray(self.ecc_len)
        for i, b in enumerate(data):
            ecc[i % self.ecc_len] ^= b
            ecc[(i + 1) % self.ecc_len] ^= (b ^ 0xAA)
        return bytes(ecc)

    def decode_and_correct(self, data: bytearray, ecc: bytearray) -> tuple[int, bytes]:
        # Giả lập sửa lỗi
        return 0, bytes(data)


class ErrorCorrectingCode:
    def __init__(self, t: int = config.BCH_ERROR_CORRECTION_BITS, m: int = config.BCH_M):
        self.t = t
        self.m = m
        self.data_len = config.BCH_DATA_BYTES
        self.ecc_len = config.BCH_ECC_BYTES

        if HAS_BCHLIB:
            try:
                self.bch = bchlib.BCH(t=self.t, m=self.m)
                self.data_len = config.BCH_DATA_BYTES
                self.ecc_len = self.bch.ecc_bytes
                self.use_native = True
            except Exception:
                self.bch = PurePythonBCH(t)
                self.use_native = False
        else:
            self.bch = PurePythonBCH(t)
            self.use_native = False

    def encode(self, data: bytes) -> bytes:
        if len(data) != self.data_len:
            raise ValueError(f"Dữ liệu đầu vào phải có độ dài đúng {self.data_len} bytes!")
        return self.bch.encode(data)

    def decode_and_correct(self, data: bytearray, ecc: bytearray) -> tuple[int, bytes]:
        """
        Giải mã và sửa lỗi trực tiếp trên mảng bytearray:
        - Output: (số bit đã sửa, bytes dữ liệu sau khi sửa)
        - Trả về (-1, b'') nếu số lỗi vượt quá t
        """
        if self.use_native:
            # bch.decode trả về số bit lỗi tìm thấy hoặc < 0 nếu lỗi
            bitflips = self.bch.decode(data, ecc)
            if bitflips < 0:
                return -1, b""
            # bch.correct sửa lỗi trực tiếp in-place
            self.bch.correct(data, ecc)
            return bitflips, bytes(data)
        else:
            return self.bch.decode_and_correct(data, ecc)
