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
SAVE_ALL = False


def process(source_file, out_path, model_path):
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
    img_orig = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    results.append(img_orig)
    titles.append('original image')
    print('Загрузили картинку размерностью: {}'.format(img_orig.shape))

    # Делаем АВТОКОНТРАСТ
    img_autocont = u.autocontrast(img_orig)
    results.append(img_autocont)
    titles.append('auto contrast')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'auto_cont_' + os.path.basename(source_file))
        cv.imwrite(out_file, img_autocont)
        print('Сохранили рез-т автоконтраста размерностью: {}'.format(img_autocont.shape))

    # Получаем модель и делаем ПРЕДИКТ МАСКИ
    model = m.get_model(model_path, VERBOSE)
    pred = m.pred_images(model, img_orig, img_height, img_width)[0]
    results.append(cv.cvtColor(pred, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('predicted mask')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'mask_predict_' + os.path.basename(source_file))
        cv.imwrite(out_file, pred)
        print('Сохранили предикт маски размерностью: {}'.format(pred.shape))

    # Наложим МАСКУ НА ИЗОБРАЖЕНИЕ
    ones = np.ones((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    img_pred = img_orig.copy()
    img_pred[:, :, 0] = np.where(pred == 255, img_pred[:, :, 0], ones[:, :, 0] * 0)
    img_pred[:, :, 1] = np.where(pred == 255, img_pred[:, :, 1], ones[:, :, 0] * 255)
    img_pred[:, :, 2] = np.where(pred == 255, img_pred[:, :, 2], ones[:, :, 0] * 0)
    results.append(img_pred)
    titles.append('mask on image')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'mask_on_img_' + os.path.basename(source_file))
        cv.imwrite(out_file, img_pred)
        print('Сохранили изображение с маской размерностью: {}'.format(img_pred.shape))

    # Смотрим HSV & THRESH_BINARY
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    (ret, img_hsv_thresh) = cv.threshold(S, 32, 255, cv.THRESH_BINARY)
    results.append(cv.cvtColor(img_hsv_thresh, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('hsv -> binary')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'hsv_thresh_img_' + os.path.basename(source_file))
        cv.imwrite(out_file, img_hsv_thresh)
        print('Сохранили изображение HSV & THRESH_BINARY: {}'.format(img_hsv_thresh.shape))

    # Преобразование HSV & THRESH_BINARY -> CONTOURS
    img_contours = u.opencv_contours(img_bgr.copy())
    results.append(img_contours)
    titles.append('binary -> contours')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'contours_img_' + os.path.basename(source_file))
        cv.imwrite(out_file, img_contours)
        print('Сохранили изображение с контурами: {}'.format(img_contours.shape))

    # Добавим "заглушки" чтобы пропустить две ячейки
    img_black = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for _ in range(2):
        results.append(img_black)
        titles.append('')

    # Преобразование CANNY над ОРИГИНАЛОМ
    gray = cv.cvtColor(img_orig, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 30, 200)
    results.append(cv.cvtColor(edges, cv.COLOR_GRAY2RGB))  # результат запишем в размерности (w, h, 3)
    titles.append('canny edges')
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'canny_edges' + os.path.basename(source_file))
        cv.imwrite(out_file, edges)
        print('Сохранили изображение Canny edges: {}'.format(edges))

    # Преобразование CANNY -> CONTOURS
    gray = cv.cvtColor(img_orig, cv.COLOR_RGB2GRAY)
    #
    CANNY_THRESH_1 = 10
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
    if SAVE_ALL:
        out_file = os.path.join(out_path, 'canny_contours' + os.path.basename(source_file))
        cv.imwrite(out_file, edges)
        print('Сохранили изображение Canny contours: {}'.format(edges))








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



