import sqlite3
import json

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

# Показываем пользователей
cursor.execute("SELECT id, name FROM users")
users = cursor.fetchall()

print("Пользователи:")
for user in users:
    print(user)

# Показываем количество векторов
cursor.execute("SELECT COUNT(*) FROM face_vectors")
count = cursor.fetchone()[0]

print(f"\nВсего векторов: {count}")

# Проверяем размер первого вектора
cursor.execute("SELECT vector FROM face_vectors LIMIT 1")
row = cursor.fetchone()

if row:
    vector = json.loads(row[0])
    print(f"Размер первого вектора: {len(vector)}")

conn.close()