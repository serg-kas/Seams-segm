"""
Исходное изображение обрабатываем детектором краев Canny и нейросетью.
Пробуем совместить результаты Canny и нейросети.
Результаты визуализируем и сохраняем как выходной файл.
Исходная картинка подается в программу как параметр или берется случайно из папки imgs_fs.
"""
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
IMG_SIZE = 512


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
    print('Загрузили картинку размерами: ({},{})'.format(img_orig.shape[0], img_orig.shape[1]))

    # Делаем предикт моделью


    # tmp = cv.resize(img_orig, (900, 500), interpolation=cv.INTER_AREA)
    # cv.imshow('original image', tmp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Делаем предикт моделю


    # Получаем модель
    # model = m.get_model(model_path, VERBOSE)





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



