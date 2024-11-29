from TArtnet import StupidArtnet
import time

# 初始化設定
target_ip = '169.254.44.100'  # 目標 IP
universe = 0                  # Universe 編號
packet_size = 512             # 資料包大小
frame_rate = 40               # 更新頻率 (Hz)

# 創建 StupidArtnet 物件
artnet = StupidArtnet(target_ip, universe, packet_size, frame_rate, True, True)

# 檢查初始化
print(artnet)

# 定義燈光資料列表，每個元素是一個長度為 packet_size 的 bytearray
data_list = [
    bytearray([255,255,255,0,0,0,0,0,0,0
               ,0,0,0,0,0,0,50,255,255,255
               ,0,255,255,0,0,0,0,0,0,0
               ,0,0,0,0,85,255,0,0,0,0
               ,0,0,0,0,0,0,0,0,0,0
               ,85,255,0,0,0,0,0,0,0,0
               ,0,0,0,0,0,0,85,255,0,0
               ])
]

try:
    # 開始發送
    artnet.start()
    
    while True:
        for data in data_list:
            artnet.set(data)  # 設定燈光資料
            artnet.show()
            time.sleep(1 / frame_rate)  # 等待以維持 40Hz 頻率

except KeyboardInterrupt:
    # 停止發送，清除燈光
    print("\nStopping Art-Net transmission...")
    artnet.blackout()
    artnet.stop()
    del artnet
    print("Finished.")
