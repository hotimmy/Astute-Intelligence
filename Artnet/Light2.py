from TArtnet import StupidArtnet
import time
import sys
import select

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
packet = bytearray([0] * packet_size)

# 非阻塞輸入檢查函數
def check_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

try:
    # 開始發送
    artnet.start()
    artnet.set(packet)
    i = 0  # 初始化變數
    direction = 10  # 控制遞增或遞減
    fade_mode = False  # 狀態變數，控制是否執行漸變

    while True:
        user_input = check_input()
        
        if user_input == "1":
            print("PN on")
            artnet.set_single_value(17, 63)
            artnet.set_single_value(18, 255)
            artnet.set_single_value(19, 255)
            artnet.set_single_value(20, 255)


        elif user_input == "2":
            print("PN off")
            artnet.set_single_value(17, 0)
            artnet.set_single_value(18, 0)
            artnet.set_single_value(19, 0)
            artnet.set_single_value(20, 0)


        elif user_input == "3":
            print("color default")
            artnet.set_single_value(50, 251)
            artnet.set_single_value(51, 179)
            artnet.set_single_value(52, 35)


        elif user_input == "4":
            print("P on")
            artnet.set_single_value(49, 255)
            fade_mode = False  # 停止漸變

        elif user_input == "5":
            print("啟動漸變效果")
            fade_mode = True  # 啟用漸變

        # 如果漸變模式啟用，執行漸變邏輯
        if fade_mode:
            i += direction
            if i >= 255:
                direction = -10  # 開始遞減
            elif i <= 0:
                direction = 10  # 開始遞增
            artnet.set_single_value(49, i)
            

        # 保持40Hz頻率
        artnet.show()
        time.sleep(1 / frame_rate)

except KeyboardInterrupt:
    # 停止發送，清除燈光
    print("\nStopping Art-Net transmission...")
    artnet.blackout()
    artnet.stop()
    del artnet
    print("Finished.")
