import tensorflow as tf

# 載入模型（即使使用舊版格式）
model = tf.keras.models.load_model('/Users/timmy/Documents/GitHub/Astute-Music/Artnet/tempmodel2024113017352.keras')

# 保存為新的 Keras 3 支援的格式
model.save('/Users/timmy/Documents/GitHub/Astute-Music/Artnet/converted_model.keras')
