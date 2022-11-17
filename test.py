import matplotlib.image as mpigm
import matplotlib.pyplot as plt
import numpy as np

def reduce(img, factor):
    # двумерный массив, заполненный нулями, соответствует размерности изображения
    # img.shape[0] - количество пикселей
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])

    return result


img2=mpigm.imread("test.jpg")
# img = reduce(img2,2)


l=[[0 for i in range(3)] for j in range(3)]
x = np.array([[1,2],[3,4]])
l=np.array(l)
print(x*2)

a=[[0,1,]]
print


