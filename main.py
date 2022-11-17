import numpy as np
import matplotlib.image as mpigm
import matplotlib.pyplot as plt
from rangTree import *
# разделение на ранги*****************

# получаем черно-белое изображение
def get_greyscale_image(img):
    #mean - average for massive
    return np.mean(img[:,:,:2], 2)
#изменение размеров блоков
def reduce(img, factor):
    #двумерный массив, заполненный нулями, соответствует размерности изображения
    #img.shape[0] - количество пикселей
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])

    return result


img=mpigm.imread("test.jpg")

img = get_greyscale_image(img)


#список рангов
# range_list = [ range_struct() for i in range(4*4)]
# for i in range(len(range_list)):
#     range_list[i].arr[0]=i+1
# for range in range_list:
#     print(range.arr)

#Добавить ввод глубины дерева и ошибку

# img = reduce(img, 4)
# plt.figure()

# plt.imshow(img, cmap='gray', interpolation='none')
# plt.show()

# алгоритм преобразования
# for i in range(len(range_list)):
#     level_index=0
    #for j in range(max_tree_depth):
        #bit_index =
