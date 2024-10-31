import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Указание пути для сохранения
save_path = './clahe_laplacian/'
# Создание директории, если она не существует
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(img)

# Шаг 3: Усиление краев с использованием оператора Лапласа
laplacian = cv2.Laplacian(equalized_img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)  # Преобразование обратно в 8-битное изображение

# Совмещаем изображение с его границами для усиления контуров
edges_image = cv2.addWeighted(equalized_img, 1.0, laplacian_abs, 0.5, 0)

# Шаг 5: Отображение оригинального и обработанных изображений
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image (CLAHE)', equalized_img)
cv2.imshow('Edges', edges_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 6: Сохранение изображений
cv2.imwrite( save_path + '/equalized_image.png', equalized_img)
cv2.imwrite( save_path + '/edges_image.png', edges_image)
