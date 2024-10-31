import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Указание пути для сохранения
save_path = './clahe'
# Создание директории, если она не существует
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(img)

# Шаг 3: Отображение оригинального и обработанных изображений
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image (CLAHE)', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 4: Сохранение изображений
cv2.imwrite(save_path + '/equalized_image.png', equalized_img)