"""
Исходное изображение обрабатываем детектором краев Canny и нейросетью.
Пробуем совместить результаты Canny и нейросети.
Результаты визуализируем и сохраняем как выходной файл.
Исходная картинка подается в программу как параметр или берется случайно из папки imgs_fs.
"""
import numpy as np
import cv2 as cv
import sys
import os
import random
import utils as u
import unet as m

# Выводить дополнительную информацию
VERBOSE = True

# Допустимые форматы изображений
img_type_list = ['.jpg', '.jpeg', '.png']

# Размер к которому приводить изображение для нейросети
img_height = 512
img_width = 512


def process(source_file, out_path, model_path):
    """
    :param source_file: исходное изображение
    :param out_path: путь куда записать результат
    :param model_path: путь к НС модели
    """
    # Создадим папку для выходных файлов, если ее нет
    if not (out_path in os.listdir('.')):
        os.mkdir(out_path)

    # Объявим список куда будем сохранять результаты работы
    results_list = []

    # Загружаем исходное изображение
    img_orig = cv.imread(source_file)
    img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
    print('Загрузили картинку размерностью: {}'.format(img_orig.shape))

    # Получаем модель и делаем предикт
    model = m.get_model(model_path, VERBOSE)
    pred = m.pred_images(model, img_orig, img_height, img_width)[0]
    print(pred.shape)

    # Нанесем маску на исходное изображение для визуализации
    ones = np.ones((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    img_pred = img_orig.copy()
    img_pred[:, :, 0] = np.where(pred == 255, img_pred[:, :, 0], ones[:, :, 0] * 0)
    img_pred[:, :, 1] = np.where(pred == 255, img_pred[:, :, 1], ones[:, :, 0] * 255)
    img_pred[:, :, 2] = np.where(pred == 255, img_pred[:, :, 2], ones[:, :, 0] * 0)

    # Смотрим что даст преобразование Canny
    img_canny = u.opencv_canny(u.img_resize_cv(img_orig))



    from PIL import Image
    Image.fromarray(img_canny).show()









if __name__ == '__main__':
    """
    В программу можно передать параметры:
    sys.argv[1] - source_path - путь к папке или отдельному файлу для обработки
    sys.argv[2] - out_path - путь к папке для сохранения результата
    sys.argv[3] - model_path - путь к модели 
    """
    source_path = 'imgs_fs' if len(sys.argv) <= 1 else sys.argv[1]
    out_path = 'out_files' if len(sys.argv) <= 2 else sys.argv[2]
    model_path = 'models/00-Unet' if len(sys.argv) <= 3 else sys.argv[3]
    #
    if os.path.isdir(source_path):
        source_files = os.listdir(source_path)
        if len(source_files) == 0:
            print('Папка {} для исходных файлов пуста'.format(source_path))
        else:
            random_file = random.choice(source_files)
            print('Выбрали случайно: {}'.format(random_file))
            process(os.path.join(source_path, random_file), out_path, model_path)
    elif os.path.isfile(source_path):
        source_file = source_path
        print('Обрабатываем: {}'.format(source_file))
        process(source_file, out_path, model_path)
    else:
        print("Не нашли данных для обработки.")



