# Dinamik model yükleme dışında her şey sabit, sadece hatalı kodlar temizlenmiş app.py oluşturuluyor
with open("/mnt/data/app.py", "r", encoding="utf-8") as f:
    raw_code = f.read()

# Gereksiz "with open(...)" parçalarını temizle
cleaned_code = "\n".join([
    line for line in raw_code.splitlines()
    if not line.strip().startswith("path =") and "open(" not in line
])

# Dosyayı tekrar kaydet
with open("/mnt/data/app.py", "w", encoding="utf-8") as f:
    f.write(cleaned_code)

"/mnt/data/app.py"
