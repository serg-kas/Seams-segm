# Seams-segm

Сегментация изображения по двум классам: 
 - плитка(кирпич,доска)
 - швы 
 
 23/24-08
 Файлы изображений для обучения приведены к размеру 1024 по максимальной стороне.
 Маски (представляли из себя на самом деле картинки jpg) были обработаны, почищены и приведены к двум классам. 
 Предобработка см. в utils.py функция imgs_preparing.
 
 Пришлось ограничиться 41 файлом изображений. Ввиду небольшого размера и количества файлов пока таскаю их через github.
 
 
 24/25-08
 Для обучения переходим в Google Colab.
 
 Сделал бесконечный (с аугментацией) генератор изображений и ряд вспомогательных функций (метрики, визуализация и т.д.)
 
 1. Unet - СДЕЛАНО. Результаты см.в ноутбуке Model_Unet.jpynb
 
 2. PSP-net - ОТМЕНА ввиду хороших результатов унетки. 
 
 
 27-29-08
 
 1. Доучил унетку на имеющимся наборе данных (код в любом случае пригодится). Плюс повторил еще раз с изменением аугментации. 

 2. Сделал вариант дата-генератора, использующего полное разрешение исходных картинок и обучил модель на нём.
 3. 
 Результаты:  
 - если учим с нуля, то плохо. Картинку в целом модель потом не воспринимает.
 - если учим с весов первой унетки (которую учили на ресайце полных фото), то результаты лучше. Учим конечно же малым lr.
 
 
 Далее:
 Датасет надо расширять. Без этого все равно дальше не двинуться.
 Есть такой метод opencv (детекция краёв).
 
 
 
 
 
 
 




