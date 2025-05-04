# Güncel app.py dosyasına dinamik model yüklemeyi ekle
path = "/mnt/data/app.py"

# Dosyayı oku
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Sabit model yükleme satırını bul ve değiştir
old_line = 'model = tf.keras.models.load_model("mnist_model.h5")'
new_lines = '''
model_path = "updated_model.h5" if os.path.exists("updated_model.h5") else "mnist_model.h5"
model = tf.keras.models.load_model(model_path)
st.markdown(f"📦 Kullanılan model: `{model_path}`")
'''

if old_line in code:
    code = code.replace(old_line, new_lines)

# Dosyayı güncelle
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

path
