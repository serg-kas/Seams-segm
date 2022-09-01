"""
Исходное изображение обрабатываем детектором краев Canny и нейросетью.
Пробуем совместить результаты Canny и нейросети.
Результаты визуализируем и сохраняем как выходной файл.
Исходная картинка подается в как параметр или берется случайно из папки imgs_fs.
"""
# Модуль с функциями
import utils as u
# Прочее
import sys
import os
import random
import utils as u
import unet as m

# Выводить дополнительную информацию
VERBOSE = True

# Допустимые форматы изображений
img_type_list = ['.jpg', '.jpeg', '.png']

# Размер к которому приводить изображение
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

    # Получаем модель
    model = m.get_model(model_path)

    # Загружаем исходное изобра



if __name__ == '__main__':
    source_path = 'imgs_fs' if len(sys.argv) <= 1 else sys.argv[1]
    out_path = 'out_files' if len(sys.argv) <= 2 else sys.argv[2]
    model_path = 'models/00-Unet' if len(sys.argv) <= 3 else sys.argv[2]
    #
    if os.path.isdir(source_path):
        source_files = os.listdir(source_path)
        assert len(source_files) > 0, "Папка с исходными файлами пуста"
        random_file = random.choice(source_files)
        print('Обрабатываем: {}'.format(random_file))
        process(random_file, out_path, model_path)
    elif os.path.isfile(source_path):
        source_file = source_path
        print('Обрабатываем: {}'.format(source_file))
        process(source_file, out_path, model_path)
    else:
        print("Не нашли файла для обработки.")



