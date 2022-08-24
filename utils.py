# Модуль функций
import numpy as np
import cv2 as cv
import random
import math
import os
import sys
import time


# Функция автокоррекции контраста
def autocontrast(img):
    # converting to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    result = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return result


# Функция проверки нужен ли этот файл для обработки
# (специфически для данного датасета)
def file_check(file_name, file_extension, img_type_list):
    black_list = ['height',
                  'BrickRedSmall1752_4K_mask',
                  'BricksRedOld_7028_4K_mask',
                  'Bricks_grey_4927_diffuse_8K',
                  'Geomesh_1549_diffuse_8K',
                  'Paving_brown_square_4316_diffuse_8K',
                  'Paving_fishscale_5198_diffuse_8K',
                  'Paving_grainy_flat_3180_diffuse_8K',
                  'Paving_grassy_6623_diffuse_8K',
                  'Paving_long_brown_4317_diffuse_8K',
                  'Paving_origami_6931_diffuse_8K',
                  'Paving_rounded_red_6174_diffuse_8K',
                  'Paving_simple_colored_7183_diffuse_8K',
                  'Paving_simple_snowy_3387_diffuse_8K',
                  'Paving_snowy_3215_diffuse_8K',
                  'Paving_trapeze_9174_diffuse_8K',
                  'Paving_trapeze_mini_3387_diffuse_8K',
                  'Roofing_corrida_8762_diffuse_4K',
                  'Shingles_Hex_8174_diffuse',
                  'Stone_blocks_6371_diffuse_16K',
                  'Travertin_yellow_7349_diffuse_8K',
                  'Wood_planks_5102_diffuse_4K',
                  'Wood_planks_8127_diffuse_4K',
                  'Wood_planks_grey_9351_diffuse_8K',
                  'Wood_planks_grey_9352_diffuse_8K',
                  'Wood_planks_new_7543_diffuse_8K',
                  'Brick_orange_3997_diffuse_8K_dark',
                  'Paving_antique_7375_diffuse2_8K',
                  'Paving_park_4658_diffuse_kids_8K',
                  'Paving_park_wet_4659_diffuse_8K',
                  'Paving_rounded_4380_diffuse_8K',
                  'Paving_rounded_4381_diffuse_8K',
                  'Paving_salmon_4167_diffuse_8K',
                  'Paving_salmon_4167_diffuse_grey_8K',
                  'Paving_tactile_mini_7145_diffuse2_8K',
                  'Paving_tactile_mini_7145_diffuse_8K',
                  'Paving_tiara_4891_diffuse_red_4K',
                  'Planks_painted_1367_diffuse3_8K',
                  'Planks_painted_1367_diffuse_8K',
                  'Rubber_tiles_1978_Diffuse2_4K',
                  'Rubber_tiles_1978_Diffuse_4K',
                  'Shinglas_brown_3176_diffuse_8K',
                  'Tactile_Paving_6872_diffuse_4K',
                  'Tactile_Paving_6872_mask_4K',
                  ]
    if file_extension not in img_type_list:
        return False
    for word in black_list:
        if word.lower() in file_name.lower():
            return False
    return True


# Функция подготовки изображений
def imgs_preparing(source_path, out_path, img_type_list, img_size=1024, crop=0.0, verbose=False):

    if verbose:
        time_start = time.time()

    # Создадим папку для файлов, если её нет
    if not (out_path in os.listdir('.')):
        os.mkdir(out_path)

    # Создадим список файлов картинок для обработки
    source_files = sorted(os.listdir(source_path))
    img_files = []
    mask_count = 0 # Попутно посчитаем сколько у нас файлов с масками
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # Проверяем отдельной функцией брать ли файл в датасет
        if file_check(filename, file_extension, img_type_list):
            img_files.append(f)
            if 'mask' in filename.lower():
                mask_count += 1

    if verbose:
        print('Найдено файлов с масками: {}'.format(mask_count))
        print('Всего к обработке файлов: {}'.format(len(img_files)))

    # Обрабатываем
    for file in img_files:
        # полные пути к файлам
        img_file = os.path.join(source_path, file)
        out_file = os.path.join(out_path, file)
        # Загружаем изображение
        img = cv.imread(img_file)

        # Размеры картинки
        height = img.shape[0]
        width = img.shape[1]

        # Обрезаем картинку по краям
        if crop > 0:
            assert crop < 1
            crop_h =int(crop * height)
            crop_w = int(crop * width)
            img = img[crop_h:height-crop_h, crop_w:width-crop_w]
            # Новый размеры картинки
            height = img.shape[0]
            width = img.shape[1]
            # print('Размер картинки ПОСЛЕ кропа {}'.format(img.shape))

        # Рассчитаем коэффициент для изменения размера
        if width > height:
            scale_img = img_size / width
        else:
            scale_img = img_size / height
        # и целевые размеры изображения
        new_width = int(width * scale_img)
        new_height = int(height * scale_img)

        # делаем автокоррекцию контраста
        # img = autocontrast(img)

        # делаем ресайз
        img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)

        # cv.imshow('image', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        try:
            cv.imwrite(out_file, img)
        except IOError:
            print('Не удалось сохранить файл: {}'.format(out_file))
        finally:
            if verbose:
                print('Обработали файл: {}'.format(out_file))

    if verbose:
        time_end = time.time() - time_start
        print('Время обработки, сек: {0:.1f}'.format(time_end))







