import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Генерируем тестовые данные и сохраняем их в файл data.csv

laboratory_work_list = []
laboratory_work = ("удовлетворительно", "хорошо", "отлично")
grant = np.random.randint(0, 100, (2500, 6))
student_scores = pd.DataFrame (grant, columns = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6'])
#for i in range (2000): laboratory_work_list.insert (i, np.random.choice(laboratory_work))
student_scores_avg = pd.DataFrame(student_scores.mean(axis=1))
a = student_scores_avg.to_numpy()
for i in range (2500):
    if a[i] >= 70: laboratory_work_list.insert (i, laboratory_work[2])
    if a[i] < 70 and a[i] >= 50: laboratory_work_list.insert (i, laboratory_work[1])
    if a[i] < 50 and a[i] >= 30: laboratory_work_list.insert (i, laboratory_work [0])
    if a[i] < 30: laboratory_work_list.insert (i, " ")
student_scores['laboratory_work'] = laboratory_work_list
student_scores = student_scores.dropna()
student_scores.to_csv('test.csv', index=False)

# Загружаем данные из файла data.csv
student_scores = pd.read_csv('test.csv')

# Преобразуем категориальные признаки в числовые с помощью Label Encoding
cat_features = ['subject1', 'subject2', 'subject3']
for feature in cat_features:
    student_scores[feature] = student_scores[feature].astype('category')

# Разделяем данные на обучающий и тестовый наборы
X = student_scores.drop('laboratory_work', axis=1)
y = student_scores['laboratory_work']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель CatBoostClassifier
model = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.1, cat_features=cat_features)
model.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе
y_pred = model.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

def forecast_score (subject1, subject2, subject3, subject4, subject5, subject6):

  return f"Точность модели: {accuracy}"

#Необходимо указать оценки студента (от 0 до 99) по 6 предметам
#forecast_score (10, 20, 30, 40, 50, 60)


