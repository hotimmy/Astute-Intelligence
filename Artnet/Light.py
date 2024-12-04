from TArtnet import StupidArtnet
from SignalGenerator import SignalGenerator as sg
import time
import sys
import select

# 初始化設定
target_ip = '10.113.4.3'  # 目標 IP
universe = 0                  # Universe 編號
packet_size = 512             # 資料包大小
frame_rate = 40               # 更新頻率 (Hz)

BPM = 120

sg = sg()

# 創建 StupidArtnet 物件
artnet = StupidArtnet(target_ip, universe, packet_size, frame_rate, True, True)

# 檢查初始化
print(artnet)

# 定義燈光資料列表，每個元素是一個長度為 packet_size 的 bytearray
packet = bytearray([0] * packet_size)

#定義燈具
B1 = 1
B2 = 2
B3 = 3
PN1 = 17
PN2 = 19
P1 = 33
P2 = 49
P3 = 65
P4 = 81
P5 = 97
P6 = 113


# 非阻塞輸入檢查函數
def check_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

try:
    # 開始發送
    artnet.start()
    artnet.set(packet)

    effect = False

    while True:
        user_input = check_input()
        
        if user_input =="bon":
            for i in [B1,B2,B3]:
                artnet.set_single_value(i,255)

        elif user_input =="boff":
            for i in [B1,B2,B3]:
                artnet.set_single_value(i,0)


        elif user_input == "pnon":
            print("PN on")
            artnet.set_single_value(PN1, 63)
            artnet.set_single_value(PN1+1, 255)
            artnet.set_single_value(PN2, 255)
            artnet.set_single_value(PN2+1, 255)


        elif user_input == "pnoff":
            print("PN off")
            artnet.set_single_value(PN1, 0)
            artnet.set_single_value(PN1+1, 0)
            artnet.set_single_value(PN2, 0)
            artnet.set_single_value(PN2+1, 0)


        elif user_input == "color":
            print("color default")
            for i in [P1,P2,P3,P4,P5,P6]:
                artnet.set_single_value(i+1, 251)
                artnet.set_single_value(i+2, 179)
                artnet.set_single_value(i+3, 35)


        elif user_input == "7":
            print("P on")
            for i in [P1,P2,P3,P4,P5,P6]:
                artnet.set_single_value(i, 255)
            effect = False

        elif user_input == "8":
            print("P off")
            for i in [P1,P2,P3,P4,P5,P6]:
                artnet.set_single_value(i, 0)
            effect = False

        elif user_input == "0":
            #漸進漸出 bpm
            sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
            sg.set_generator(P2, "sin", BPM, frame_rate, offset=0.2)
            sg.set_generator(P3, "sin", BPM, frame_rate, offset=0.4)
            sg.set_generator(P4, "sin", BPM, frame_rate, offset=0.6)
            sg.set_generator(P5, "sin", BPM, frame_rate, offset=0.8)
            sg.set_generator(P6, "sin", BPM, frame_rate, offset=1) 
            effect = True

        elif user_input == "1":
            #Can-Can bpm
            sg.set_generator(P1, "square", BPM, frame_rate, offset=0)
            sg.set_generator(P2, "square", BPM, frame_rate, offset=0)
            sg.set_generator(P4, "square", BPM, frame_rate, offset=0)
            sg.set_generator(P3, "square", BPM, frame_rate, offset=60/BPM)
            sg.set_generator(P5, "square", BPM, frame_rate, offset=60/BPM)
            sg.set_generator(P6, "square", BPM, frame_rate, offset=60/BPM)
            effect = True

        elif user_input == "2":
            #Can-Can 柔 bpm
            sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
            sg.set_generator(P2, "sin", BPM, frame_rate, offset=0)
            sg.set_generator(P3, "sin", BPM, frame_rate, offset=0)
            sg.set_generator(P4, "sin", BPM, frame_rate, offset=30/BPM)
            sg.set_generator(P5, "sin", BPM, frame_rate, offset=30/BPM)
            sg.set_generator(P6, "sin", BPM, frame_rate, offset=30/BPM) 
            effect = True

        elif user_input == "3":
            #Rain bpm
            sg.set_generator(P1, "rain", BPM, frame_rate, offset=0)
            sg.set_generator(P2, "rain", BPM, frame_rate, offset=BPM/30/6*1 )
            sg.set_generator(P3, "rain", BPM, frame_rate, offset=BPM/30/6*3 )
            sg.set_generator(P4, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
            sg.set_generator(P5, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
            sg.set_generator(P6, "rain", BPM, frame_rate, offset=BPM/30/6*5 ) 
            effect = True

        elif user_input == "4":
            #快閃
            flash = 600
            sg.set_generator(P1, "square", flash, frame_rate, offset=0)
            sg.set_generator(P2, "square", flash, frame_rate, offset=0.1)
            sg.set_generator(P3, "square", flash, frame_rate, offset=0.2)
            sg.set_generator(P4, "square", flash, frame_rate, offset=0.3)
            sg.set_generator(P5, "square", flash, frame_rate, offset=0.4)
            sg.set_generator(P6, "square", flash, frame_rate, offset=0.5) 
            effect = True
        
        elif user_input == "5":
            #漸進漸出 慢
            slow = 60
            sg.set_generator(P1, "sin", slow, frame_rate, offset=0)
            sg.set_generator(P2, "sin", slow, frame_rate, offset=0.2)
            sg.set_generator(P3, "sin", slow, frame_rate, offset=0.4)
            sg.set_generator(P4, "sin", slow, frame_rate, offset=0.6)
            sg.set_generator(P5, "sin", slow, frame_rate, offset=0.8)
            sg.set_generator(P6, "sin", slow, frame_rate, offset=1) 
            effect = True

        elif user_input == "6":
            #漸進漸出 快
            fast = 120
            sg.set_generator(P1, "sin", fast, frame_rate, offset=0)
            sg.set_generator(P2, "sin", fast, frame_rate, offset=0.2)
            sg.set_generator(P3, "sin", fast, frame_rate, offset=0.4)
            sg.set_generator(P4, "sin", fast, frame_rate, offset=0.6)
            sg.set_generator(P5, "sin", fast, frame_rate, offset=0.8)
            sg.set_generator(P6, "sin", fast, frame_rate, offset=1) 
            effect = True
        
        if effect == True:
            for i in [P1,P2,P3,P4,P5,P6]:
                artnet.set_single_value(i,sg.signal_generator(i))
        
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
