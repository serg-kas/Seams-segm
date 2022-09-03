# Модуль функций
import numpy as np
import cv2 as cv
import os
import random
import time
import re
import tkinter as tk  # используем чтобы узнать разрешение экрана
from PIL import Image


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
def imgs_preparing(source_path, imgs_path, masks_path, img_type_list, img_size=1024, crop=0.0, verbose=False):

    if verbose:
        time_start = time.time()

    # Создадим папки для файлов, если их нет
    if not (imgs_path in os.listdir('.')):
        os.mkdir(imgs_path)
    if not (masks_path in os.listdir('.')):
        os.mkdir(masks_path)

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_path))
    files_list = []
    mask_count = 0  # попутно посчитаем сколько у нас файлов с масками
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # Проверяем отдельной функцией брать или не брать файл в датасет
        if file_check(filename, file_extension, img_type_list):
            files_list.append(f)
            if 'mask' in filename.lower():
                mask_count += 1

    if verbose:
        print('Найдено файлов с масками: {}'.format(mask_count))
        print('Всего к обработке файлов: {}'.format(len(files_list)))

    # Обрабатываем
    for file in files_list:
        # полные пути к файлам
        in_file = os.path.join(source_path, file)
        # файлы переименуем по цифровым комбинациям в их именах
        filename, file_extension = os.path.splitext(file)
        filename = re.findall(r'\d+', filename)[0]
        #
        if 'mask' in file.lower():
            # out_file = os.path.join(masks_path, filename+file_extension)
            out_file = os.path.join(masks_path, filename + '.png')
            # Загружаем изображение
            img = cv.imread(in_file, 0)
        else:
            # out_file = os.path.join(imgs_path, filename+file_extension)
            out_file = os.path.join(imgs_path, filename + '.png')
            # Загружаем изображение
            img = cv.imread(in_file)

        # Размеры картинки
        height = img.shape[0]
        width = img.shape[1]

        # Обрезаем картинку по краям
        if crop > 0:
            assert crop < 1
            crop_h =int(crop * height)
            crop_w = int(crop * width)
            img = img[crop_h:height-crop_h, crop_w:width-crop_w]
            # Новые размеры картинки
            height = img.shape[0]
            width = img.shape[1]
            # print('Размер картинки ПОСЛЕ кропа {}'.format(img.shape))

        # Рассчитаем коэффициент для изменения размера
        if width > height:
            scale_img = img_size / width
        else:
            scale_img = img_size / height
        # и целевые размеры изображения
        target_width = int(width * scale_img)
        target_height = int(height * scale_img)
        # делаем ресайз
        img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_AREA)

        # Обрабатываем маску
        if 'mask' in file.lower():
            # переводим в ч/б
            # img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

            # морфология
            # kernel = np.ones((3, 3), np.uint8)
            # img = cv.dilate(img, kernel, iterations=1)
            # img = cv.erode(img, kernel, iterations=1)  # делалось для Unet-1

            # делаем трешхолд
            img = np.where(img > 200, 255, 0)

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


# Функция подготовки изображений в оригинальных размерах
def imgs_preparing_fs(source_path, imgs_path, masks_path, img_type_list, crop=0.0, verbose=False):

    if verbose:
        time_start = time.time()

    # Создадим папки для файлов, если их нет
    if not (imgs_path in os.listdir('.')):
        os.mkdir(imgs_path)
    if not (masks_path in os.listdir('.')):
        os.mkdir(masks_path)

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_path))
    files_list = []
    mask_count = 0  # попутно посчитаем сколько у нас файлов с масками
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # Проверяем отдельной функцией брать или не брать файл в датасет
        if file_check(filename, file_extension, img_type_list):
            files_list.append(f)
            if 'mask' in filename.lower():
                mask_count += 1

    if verbose:
        print('Найдено файлов с масками: {}'.format(mask_count))
        print('Всего к обработке файлов: {}'.format(len(files_list)))

    # Обрабатываем
    for file in files_list:
        # полные пути к файлам
        in_file = os.path.join(source_path, file)
        # файлы переименуем по цифровым комбинациям в их именах
        filename, file_extension = os.path.splitext(file)
        filename = re.findall(r'\d+', filename)[0]
        #
        if 'mask' in file.lower():
            # out_file = os.path.join(masks_path, filename+file_extension)
            out_file = os.path.join(masks_path, filename + '.png')
            # Загружаем изображение
            img = cv.imread(in_file, 0)
        else:
            # out_file = os.path.join(imgs_path, filename+file_extension)
            out_file = os.path.join(imgs_path, filename + '.png')
            # Загружаем изображение
            img = cv.imread(in_file)

        # Размеры картинки
        height = img.shape[0]
        width = img.shape[1]

        # Обрезаем картинку по краям
        if crop > 0:
            assert crop < 1
            crop_h =int(crop * height)
            crop_w = int(crop * width)
            img = img[crop_h:height-crop_h, crop_w:width-crop_w]
            # Новые размеры картинки
            height = img.shape[0]
            width = img.shape[1]
            # print('Размер картинки ПОСЛЕ кропа {}'.format(img.shape))

        # Обрабатываем маску
        if 'mask' in file.lower():
            # переводим в ч/б
            # img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

            # морфология
            # kernel = np.ones((3, 3), np.uint8)
            # img = cv.dilate(img, kernel, iterations=1)
            # img = cv.erode(img, kernel, iterations=1)  # делалось для Unet-1

            # делаем трешхолд
            img = np.where(img > 200, 255, 0)

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


# Функция преобразования аннотации в one hot encoding
def mask_to_ohe(ann_image, classes=[0, 255]):
    ones = np.ones((ann_image.shape[0], ann_image.shape[1], len(classes)), dtype=np.uint8)
    zeros = np.zeros((ann_image.shape[0], ann_image.shape[1], len(classes)), dtype=np.uint8)

    result = zeros.copy()

    result[:, :, 0] = np.where(ann_image == 0, ones[:, :, 0], zeros[:, :, 0])
    result[:, :, 1] = np.where(ann_image == 255, ones[:, :, 1], zeros[:, :, 1])

    return result


# Функция преобразования аннотации из ohe в классы
def ohe_to_mask(ann_ohe, classes=[0, 255]):
    ones = np.ones((ann_ohe.shape[0], ann_ohe.shape[1]), dtype=np.uint8)
    zeros = np.zeros((ann_ohe.shape[0], ann_ohe.shape[1]), dtype=np.uint8)

    result = zeros.copy()

    result = np.where(ann_ohe[:, :, 0] == 1, ones * 0, result)
    result = np.where(ann_ohe[:, :, 1] == 1, ones * 255, result)

    return result


# Функция ресайза картинки через opencv
def img_resize_cv(image, img_size=1024):
    """
    :param image: исходное изображение
    :param img_size: размер к которому приводить изображение
    :return: изображение после ресайза
    """
    curr_w = image.shape[1]
    curr_h = image.shape[0]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = img_size / curr_w
    else:
        scale_img = img_size / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image


# Функция визуализации
def show_results(results, titles, N_cols=3, N_rows=2):

    # Определим разрешение экрана
    def get_screen_resolution():
        root = tk.Tk()
        W = root.winfo_screenwidth()
        H = root.winfo_screenheight()
        print('Screen resolution: ({},{})'.format(W, H))
        if W < 1024 or H < 768:
            W = 1200
            H = 800
        return W, H

    # Get optimal font scale
    def get_optimal_font_scale(text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=scale / 10, thickness=3)
            new_width = textSize[0][0]
            if new_width <= width:
                return scale / 10
        return 1

    # Concatenation list of arrays
    def concat_from_list(frame_list, N_cols, N_rows):
        row_list = []
        for r in range(N_rows):
            row_list.append(np.concatenate(frame_list[N_cols * r: N_cols * (r + 1)], axis=1))
        return np.concatenate(row_list, axis=0)

    # Подготовим темплейт N_rows * N_cols
    # Узнаем разрешение экрана
    W, H = get_screen_resolution()
    # Размер для вывода на экран
    W, H = int(W * 0.85), int(H * 0.85)
    # Размер ячейки
    w, h = int(W / N_cols), int(H / N_rows)
    # Всего ячеек
    N_cells = int(N_cols * N_rows)
    # Занятых ячеек
    N_images = len(results)
    assert N_cells >= N_images, 'Не хватает ячеек для изображений'
    img_black = np.zeros((h, w, 3), dtype=np.uint8)
    img_black_list = [img_black for _ in range(N_cells - N_images)]
    # Готовим список картинок для полного кадра
    images_list = [cv.resize(results[idx], (w, h), interpolation=cv.INTER_AREA) for idx in range(N_images)] + img_black_list
    # Нанесем надписи
    font = cv.FONT_HERSHEY_SIMPLEX
    for idx in range(N_images):
        fontScale = get_optimal_font_scale(titles[idx], int(w * 2 / 3))
        cv.putText(images_list[idx], titles[idx], (40, 40), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)
    # Собираем и возвращаем полный кадр
    return concat_from_list(images_list, N_cols, N_rows)


# Функция canny
def opencv_canny(img):
    # To hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Get the Saturation out
    S = hsv[:, :, 1]

    # Threshold it
    (ret, T) = cv.threshold(S, 42, 255, cv.THRESH_BINARY)

    # Show intermediate result
    Image.fromarray(T).show()

    # Find contours
    contours, h = cv.findContours(T, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # img2 = img.copy()

    for c in contours:
        area = cv.contourArea(c)
        # Only if the area is not miniscule (arbitrary)
        if area > 100:
            (x, y, w, h) = cv.boundingRect(c)

            # Uncomment if you want to draw the contours
            # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

            # Get random color for each brick
            tpl = tuple([random.randint(0, 255) for _ in range(3)])
            cv.rectangle(img, (x, y), (x + w, y + h), tpl, -1)

    Image.fromarray(img).show()

    return img




