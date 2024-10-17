import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Veri setini okuyalım
df = pd.read_csv("cleaned_pavyonlar.csv")

# Veri seti hakkında bilgi alalım
print(df.info())  # Veri setinin yapısı hakkında bilgi verir
print(df.head())  # İlk 5 veriyi gösterir

# Veri ön işleme

# Price Range sütununu sayısal hale getirelim
df['Price Range'] = (
    df['Price Range']
    .str.replace('₺', '', regex=False)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

# Eksik verileri ortalama ile dolduralım
df['Price Range'] = df['Price Range'].fillna(df['Price Range'].mean())

# En sık görülen Opening Time değerini bulalım ve dolduralım
most_frequent_time = df['Opening Time'].mode()[0]
df['Opening Time'] = df['Opening Time'].fillna(most_frequent_time)

# Service Type sütunundaki NaN değerleri doldurma
df['Service Type'] = df['Service Type'].fillna('İçeride servis')

# Adres sütununu silme
df.drop(columns=['Address'], inplace=True)

# Label Encoding uygulama
label_encoder = LabelEncoder()

# Service Type sütununu Label Encoding ile dönüştür
df['Service Type'] = label_encoder.fit_transform(df['Service Type'])

# Opening Time sütununu Label Encoding ile dönüştür
opening_time_encoder = LabelEncoder()
df['Opening Time'] = opening_time_encoder.fit_transform(df['Opening Time'])

# Club Name sütununu Label Encoding ile dönüştür
df['Club Name'] = label_encoder.fit_transform(df['Club Name'])

# Sonuçları kontrol edelim
print(df.head())
print(df.info())

# Model Eğitme
# Özellikler ve hedef değişkeni ayırma
X = df.drop(columns=['Rating'])  # Bağımsız değişkenler
y = df['Rating']  # Bağımlı değişken

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model sonuçlarını değerlendirme
print(f'Training Score: {model.score(X_train, y_train)}')
print(f'Test Score: {model.score(X_test, y_test)}')

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans değerlendirme
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



