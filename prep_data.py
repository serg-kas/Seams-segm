# Подготовка данных для модели сегментации
import utils as u
# Выводить дополнительную информацию
VERBOSE = True

# Папки для картинок
imgs_orig_path = 'originals'
imgs_path = 'imgs'
masks_path = 'masks'
# Допустимые форматы изображений
img_type_list = ['.jpg', '.jpeg', '.png']
# Размер к которому приводить оригинальные изображения
IMG_SIZE = 1024
# Какую долю размера обрезать по краям
CROP = 0.1

if __name__ == '__main__':
    # Перезаписываем картинки с ресайзом и кропом в папку для готовых файлов
    u.imgs_preparing(imgs_orig_path, imgs_path, masks_path, img_type_list, IMG_SIZE, CROP, VERBOSE)




