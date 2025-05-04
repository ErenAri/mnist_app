import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# CSV dosyasını oku
csv_path = "feedback.csv"
if not os.path.exists(csv_path):
    print("❌ Hata: feedback.csv bulunamadı.")
    exit()

df = pd.read_csv(csv_path)

# Gerekli sütunları kontrol et
if "image_array" not in df.columns or "correct_label" not in df.columns:
    print("❌ Hata: Gerekli sütunlar eksik.")
    exit()

# Görüntüleri ve etiketleri hazırla
X = np.array(df["image_array"].apply(lambda x: np.fromstring(x.strip("[]"), sep=",")).to_list())
X = X.reshape(-1, 28, 28, 1) / 255.0
y = np.array(df["correct_label"])

print(f"✅ {len(X)} geri bildirim örneği ile yeniden eğitim başlıyor...")

# Basit bir CNN modeli
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Eğitim
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Kaydet
model.save("updated_model.h5")
print("✅ Yeni model 'updated_model.h5' olarak kaydedildi.")
