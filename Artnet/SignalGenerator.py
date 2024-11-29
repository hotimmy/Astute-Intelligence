import math
from collections import defaultdict

class SignalGenerator:
    def __init__(self):
        # 儲存每個 address 的生成器狀態
        self.generators = defaultdict(lambda: {"time": 0, "offset": 0, "BPM": 120, "HZ": 1, "signal_type": "sin"})

    def set_generator(self, address, signal_type, BPM, HZ, offset=0):
        """
        設定或更新指定 address 的生成器參數
        :param address: 信號的唯一地址標識
        :param signal_type: 信號類型（如 sin 或 square）
        :param BPM: 每分鐘節拍數，用於調整信號頻率
        :param HZ: 信號基頻（目前未使用）
        :param offset: 初始時間偏移量（秒）
        """
        self.generators[address] = {
            "time": 0,
            "offset": offset,
            "BPM": BPM,
            "HZ": HZ,
            "signal_type": signal_type
        }

    def signal_generator(self, address):
        """計算並返回下一個時刻的信號值"""
        # 確保 address 的生成器已存在
        if address not in self.generators:
            raise ValueError(f"Address {address} has not been initialized. Please set it first.")

        # 提取當前生成器的狀態
        gen = self.generators[address]
        time = gen["time"] + gen["offset"]  # 加入偏移量
        BPM = gen["BPM"]
        signal_type = gen["signal_type"]

        # 計算 sin 波頻率 (Hz) 基於 BPM
        frequency = BPM / 60  # BPM 轉換為每秒的週期數
        period = 1 / frequency  # 一個週期的時間長度 (秒)
        angular_frequency = 2 * math.pi * frequency  # 角速度 (rad/s)

        if signal_type == "sin":
            # 計算 sin 波值 (-1 ~ 1)，並映射到 0 ~ 255
            signal_value = math.sin(angular_frequency * time)
            scaled_value = int((signal_value + 1) / 2 * 255)
        elif signal_type == "square":
            # 計算當前時間在一個週期中的位置
            cycle_position = (time % period) / period
            # 根據位置決定高低電平
            scaled_value = 255 if cycle_position < 0.5 else 0
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")

        # 更新時間
        gen["time"] += 1 / 40  # 每次呼叫時間增加 1/40 秒

        return scaled_value

# 使用範例
sg = SignalGenerator()

# 初始化三個不同的 signal generator
sg.set_generator("addr1", "sin", 120, 1, offset=0)    # 120 BPM, sin 波, 無偏移
sg.set_generator("addr2", "square", 90, 1, offset=0.5)  # 90 BPM, 方波, 偏移 0.5 秒
sg.set_generator("addr3", "sin", 60, 1, offset=1)     # 60 BPM, sin 波, 偏移 1 秒

# 模擬呼叫信號輸出
print("Signal outputs:")
for _ in range(10):
    print(f"addr1 (sin): {sg.signal_generator('addr1')}")
    print(f"addr2 (square): {sg.signal_generator('addr2')}")
    print(f"addr3 (sin): {sg.signal_generator('addr3')}")
