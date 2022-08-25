# Модуль для работы с архитектурой Unet
import numpy as np
import cv2 as cv
import random
#
import utils as u
#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import backend as K
#
import albumentations as A
#
img_height = 512
img_width = 512


# Функция - генератор данных
def generate_data(batchsize):
    # Объявляем аугментацию
    transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomCrop(int(img_height * 0.5), int(img_width * 0.5), p=1),
            A.RandomCrop(int(img_height * 0.75), int(img_width * 0.75), p=1),
            A.RandomCrop(int(img_height * 0.9), int(img_width * 0.9), p=1),
        ], p=0.9),
        A.Resize(height=img_height, width=img_width),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=15, p=1),
            A.Blur(blur_limit=15, p=1),
        ], p=0.2),
    ])

    x_data = []
    y_data = []
    batchcount = 0
    while True:
        # Берем картинку
        file_name = random.choice(img_names)
        curr_image = cv.imread(file_name)

        # Делаем коррекцию контраста
        curr_image = u.autocontrast(curr_image)

        # Переходим к RGB
        curr_image = cv.cvtColor(curr_image, cv.COLOR_BGR2RGB)

        # Берем аннотацию
        curr_ann = cv.imread(os.path.join(anns_path, os.path.basename(file_name)), 0)

        # делаем аугментацию
        transformed = transform(image=curr_image, mask=curr_ann)
        transformed_image = transformed['image']
        transformed_ann = transformed['mask']

        # переходим к диапазону 0 до 1 и отправляем картинку в список
        x_data.append(transformed_image / 255.)

        # классы аннотации переводим ohe и отправляем в список
        y_data.append(u.mask_to_ohe(transformed_ann))

        batchcount += 1
        if batchcount >= batchsize:
            X = np.array(x_data, dtype='float32')
            y = np.array(y_data, dtype='float32')
            yield (X, y)
            x_data = []
            y_data = []
            batchcount = 0


# Функция метрики, обрабатывающая пересечение двух областей
# Кодировка OHE
def dice_coef(y_true, y_pred):
    # Возвращаем площадь пересечения деленную на площадь объединения двух областей
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


# Функция метрики, обрабатывающая пересечение двух областей d numpy
# Кодировка OHE
def dice_coef_np(y_true, y_pred):
    # Возвращаем площадь пересечения деленную на площадь объединения двух областей
    return (2. * np.sum(y_true * y_pred) + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)


#  Функция создания сети
#    Входные параметры:
#    - num_classes - количество классов
#    - input_shape - размерность карты сегментации
def unet(num_classes=2, input_shape=(1024, 1024, 3)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = MaxPooling2D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_4_out
    x = block_4_out

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out])  # Объединем текущий слой со слоем block_3_out
    x = Conv2D(256, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv2D(128, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])

    return model


# Функция предикта
def pred_images(model, images_list):
    """
    :param model: путь к  модели для загрузки
    :param images_list: список изображений для обрабоки (можно подать и одиночную картинку)
    :return: mask_list или mask: список обработанных изображений или одну маску
    """
    # Список для возврата из функции
    mask_list = []

    # Чтобы обработать одиночную картинку
    if type(images_list) is not list:
        images_list = [images_list]
        mask_list = None

    for curr_image in images_list:
        # сохраним оригинальные размеры картинки
        curr_w = curr_image.shape[1]
        curr_h = curr_image.shape[0]

        # делаем предобработку (коррекцию контраста)
        curr_image = u.autocontrast(curr_image)

        # переходим к RGB
        curr_image = cv.cvtColor(curr_image, cv.COLOR_BGR2RGB)

        # делаем ресайз к целевым размерам
        curr_image = cv.resize(curr_image, (img_width, img_height), interpolation=cv.INTER_AREA)

        # нормализуем
        curr_image = curr_image / 255.

        # добавляем ось
        curr_image = curr_image.reshape(1, img_height, img_width, 3)
        # img = np.expand_dims(img, axis = 0)

        # получаем предикт
        predict = model.predict(curr_image)
        pred = predict[0]
        pred = np.around(pred).astype(np.uint8)

        # пересчитываем его в маску
        mask = u.ohe_to_mask(pred)

        # ресайз маски к оригинальным размерам
        mask = cv.resize(mask, (curr_w, curr_h), interpolation=cv.INTER_NEAREST)

        if mask_list is not None:
            mask_list.append(mask)

    if mask_list is None:
        return mask
    else:
        return mask_list

