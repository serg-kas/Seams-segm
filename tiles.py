"""
Исходное изображение обрабатываем детектором краев Canny и нейросетью.
Пробуем совместить результаты Canny и нейросети.
Результаты визуализируем и сохраняем как выходной файл.
Исходная картинка подается в программу как параметр или берется случайно из папки imgs_fs.
"""
import numpy as np
import cv2 as cv
from PIL import Image
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

# Сохранять все результаты (иначе только сводную картинку)
SAVE_ALL = True

# Сколько картинок обрабатывать если берем случайно из папки
N_repeat = 1


def process(source_file, out_path, model):
    """
    :param source_file: исходное изображение
    :param out_path: путь куда записать результат
    :param model_path: путь к НС модели
    """
    # Создадим папку для выходных файлов, если ее нет
    if not (out_path in os.listdir('.')):
        os.mkdir(out_path)

    # Куда будем сохранять результаты работы
    results = []  # список картинок
    titles = []  # список названий

    # Загружаем ИСХОДНОЕ ИЗОБРАЖЕНИЕ
    img_bgr = cv.imread(source_file)
    # img_bgr = u.img_resize_cv(img_bgr, 2048)  # ЗАКОМЕНТИРОВАТЬ
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    results.append(img_rgb)
    titles.append('original image')
    print('Загрузили картинку размерностью: {}'.format(img_rgb.shape))

    # Делаем АВТОКОНТРАСТ
    img_autocont = u.autocontrast(img_rgb)
    results.append(img_autocont)
    titles.append('auto contrast')

    # Делаем ПРЕДИКТ МАСКИ
    pred = m.pred_images(model, img_rgb, img_height, img_width)[0]
    results.append(cv.cvtColor(pred, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('predicted mask')
    if SAVE_ALL:
        base_name = os.path.basename(source_file)
        filename, _ = os.path.splitext(base_name)
        out_file = os.path.join(out_path, 'mask_pred_' + filename + '.png')
        cv.imwrite(out_file, pred)
        print('Сохранили маску по предикту: {}'.format(pred.shape))

    # Наложим МАСКУ НА ИЗОБРАЖЕНИЕ
    ones = np.ones((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    img_pred = img_rgb.copy()
    img_pred[:, :, 0] = np.where(pred == 255, img_pred[:, :, 0], ones[:, :, 0] * 0)
    img_pred[:, :, 1] = np.where(pred == 255, img_pred[:, :, 1], ones[:, :, 1] * 255)
    img_pred[:, :, 2] = np.where(pred == 255, img_pred[:, :, 2], ones[:, :, 2] * 0)
    results.append(img_pred)
    titles.append('masked image-1')

    # Смотрим HSV & THRESH_BINARY
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    (ret, img_hsv_thresh) = cv.threshold(S, 32, 255, cv.THRESH_BINARY)
    results.append(cv.cvtColor(img_hsv_thresh, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('hsv -> thresh')

    # Преобразование HSV & THRESH_BINARY -> CONTOURS
    img_contours = u.opencv_contours(img_bgr.copy())
    results.append(img_contours)
    titles.append('thresh -> contours')

    # ОПРЕДЕЛЕНИЕ ЦВЕТА швов и ФИЛЬТРАЦИЯ по HSV
    # Берем изображение и маску от предикта нейронкой
    img = img_rgb.copy()
    mask = pred.copy()
    # Наложим на нулевое изображение фрагмент исходной картинки под маской == 0
    img_seam = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_seam[:, :, 0] = np.where(mask == 0, img[:, :, 0], img_seam[:, :, 0])
    img_seam[:, :, 1] = np.where(mask == 0, img[:, :, 1], img_seam[:, :, 1])
    img_seam[:, :, 2] = np.where(mask == 0, img[:, :, 2], img_seam[:, :, 2])
    #
    img_hsv = cv.cvtColor(img_seam, cv.COLOR_RGB2HSV)
    img_hsv_flat = np.reshape(img_hsv, (-1, 3))
    hsv_uniq = np.unique(img_hsv_flat, axis=0)
    # print(hsv_uniq[:5])
    hsv_uniq = hsv_uniq[1:]  # первым был [0,0,0]

    h_min = np.min(hsv_uniq[:, 0])
    h_max = np.max(hsv_uniq[:, 0])

    s_min = np.min(hsv_uniq[:, 1])
    s_max = np.max(hsv_uniq[:, 1])

    v_min = np.min(hsv_uniq[:, 2])
    v_max = np.max(hsv_uniq[:, 2])

    low_HSV = np.array([h_min, s_min, v_min])
    high_HSV = np.array([h_max, s_max, v_max])
    # print(min_HSV, high_HSV)

    # Threshold по диапазону HSV
    new_mask = cv.inRange(img_hsv, low_HSV, high_HSV)
    #
    # kernel = np.ones((3, 3), np.uint8)
    # new_mask = cv.dilate(new_mask, kernel, iterations=2)
    #
    img_res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ones = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_res[:, :, 0] = np.where(new_mask == 0, img[:, :, 0], ones[:, :, 1] * 0)
    img_res[:, :, 1] = np.where(new_mask == 0, img[:, :, 1], ones[:, :, 1] * 255)
    img_res[:, :, 2] = np.where(new_mask == 0, img[:, :, 2], ones[:, :, 1] * 0)
    results.append(cv.cvtColor(new_mask, cv.COLOR_GRAY2RGB))
    titles.append('filtered mask')
    results.append(img_res)
    titles.append('masked image-2')
    if SAVE_ALL:
        base_name = os.path.basename(source_file)
        filename, _ = os.path.splitext(base_name)
        out_file = os.path.join(out_path, 'mask_hsv_' + filename + '.png')
        cv.imwrite(out_file, new_mask)
        print('Сохранили маску по фильтру: {}'.format(new_mask.shape))

    # Преобразование CANNY над ОРИГИНАЛОМ
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 30, 200)
    results.append(cv.cvtColor(edges, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('canny edges')

    # Преобразование CANNY -> CONTOURS
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    #
    CANNY_THRESH_1 = 30
    CANNY_THRESH_2 = 200
    #
    edges = cv.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)
    #
    contours, h = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours, h = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    for c in contours:
        area = cv.contourArea(c)
        # Only if the area is not miniscule (arbitrary)
        if area > 100:
            (x, y, w, h) = cv.boundingRect(c)
            # cv.drawContours(img, [c], -1, (0, 255, 0), 2)
            # Get random color
            tpl = tuple([random.randint(0, 255) for _ in range(3)])
            cv.rectangle(edges, (x, y), (x + w, y + h), tpl, -1)
    results.append(cv.cvtColor(edges, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('canny -> contours')

    # Добавим "заглушки" чтобы пропустить ячейки
    img_black = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for _ in range(1):
        results.append(img_black)
        titles.append('')

    # Преобразование TEST
    # test = u.cut_and_canny_contour_cv(img_rgb.copy(), pred.copy())
    # #
    # results.append(test)  # результат запишем в размерности (w, h, 3)
    # titles.append('testing')
    # if SAVE_ALL:
    #     out_file = os.path.join(out_path, 'test_' + os.path.basename(source_file))
    #     cv.imwrite(out_file, edges)
    #     print('Сохранили изображение TEST: {}'.format(test))

    #################################
    # Сводная картинка с результатами
    full_image = u.show_results(results, titles, 4, 3)
    out_file = os.path.join(out_path, 'all_results_' + os.path.basename(source_file))
    # Сохраняем изображение
    result = Image.fromarray(full_image)
    result.save(out_file)
    print('Сохранили результаты работы: {}'.format(out_file))
    Image.fromarray(full_image).show()


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

    if os.path.isdir(source_path):
        source_files = os.listdir(source_path)
        if len(source_files) == 0:
            print('Папка {} для исходных файлов пуста'.format(source_path))
        else:
            # Модель грузим один раз и передаем в функцию
            model = m.get_model(model_path, VERBOSE)
            for n in range(N_repeat):
                print('Обрабатываем: {} из {}'.format(n+1, N_repeat))
                random_file = random.choice(source_files)
                print('Выбрали случайно: {}'.format(random_file))
                process(os.path.join(source_path, random_file), out_path, model)
    elif os.path.isfile(source_path):
        # Модель грузим и передаем в функцию
        model = m.get_model(model_path, VERBOSE)
        source_file = source_path
        print('Обрабатываем: {}'.format(source_file))
        process(source_file, out_path, model)
    else:
        print("Не нашли данных для обработки.")



