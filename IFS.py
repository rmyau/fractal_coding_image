import numpy as np
from scipy import ndimage
import matplotlib.image as mpigm
import matplotlib.pyplot as plt
from rangTree import *


# изменить проверку
def checkImage(img, blockSize):
    if img.get_width() % blockSize != 0:
        return False
    if img.get_height() % blockSize != 0:
        return False
    return True


class cIFS:
    def __init__(self, img_name, max_tree_depth, eps, maxLevelDomain):

        # уровень макс достигнутый в дереве
        self.maxFract = 0
        # погрешность
        self.eps = eps
        # растр изображения
        self.img = self.get_greyscale_image(mpigm.imread(img_name))
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        # размерность блоков рангов на нулевом уровне
        self.startSizeRange = self.width / 4
        self.domain_list = []
        self.levelRange = [[] for i in range(max_tree_depth)]
        self.max_tree_depth = max_tree_depth
        self.maxlevelDomain = maxLevelDomain
        # значения афинных преобразований
        self.directions = [1, -1]  # кооэфициент для отражения
        self.angles = [0, 90, 180, 270]  # углы для афинных преобразований
        self.candidates = [[direction, angle] for direction in self.directions for angle in self.angles]

    def rotate(self, img, angle):
        return ndimage.rotate(img, angle, reshape=False)

    def flip(self, img, direction):
        return img[::direction, :]

    def apply_transformation(self, img, direction, angle):
        return self.rotate(self.flip(img, direction), angle)

    def get_greyscale_image(self, img):
        # mean - average for massive
        return np.mean(img[:, :, :2], 2)

    def reduce(self, img, factor):  # img - в данном случае для домена это data
        # двумерный массив, заполненный нулями, соответствует размерности изображения
        # img.shape[0] - количество пикселей

        result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])

        return result

    # создание вектора доменов и рангов
    def get_rang(self, size, step,level,needArr):  # int,int,bool
        w = self.width
        h = self.height
        vector = []
        indexRang=0
        for y in range(0, h + 1, step):
            for x in range(0, w + 1, step):
                indexRang+=1
                # блок - экземпляр домена - хранить в списке доменов
                block = range_struct(size=size, step=step, x=x, y=y, img=self.img, level=level)
                if needArr:
                    block.arr = byte(self.max_tree_depth).arr
                    block.arr[0]=indexRang
                vector.append(block)
        return vector

    def brightness_contrast(self, rang, domain):
        size = domain.size
        dataD = np.array(domain.data)
        dataR = np.array(rang.data)
        sumD = np.sum(dataD)
        sumR = np.sum(dataR)

        d = (1 / (size * size)) * sumD
        r = (1 / (size * size)) * sumR
        b = np.sum(np.square(dataD - d))

        newD = dataD - d
        newR = dataR - r
        res = np.array([[0 for i in range(size)] for j in range(size)])
        for i in range(size):
            for j in range(size):
                res[i][j] = newD[i][j] * newR[i][j]
        a = np.sum(res)

        o = r - (a / b) * d
        s = a / b

        return (s, o)

    def compress(self, img):  # eps - погрешность
        # checkImage
        s = self.startSizeRange
        # найдем начальный уровень рангов
        self.levelRange[0] = self.get_rang(s, s, 0,True)
        # размер для домена
        sD = s * 2
        for level in range(self.maxlevelDomain):
            self.domain_list.append(self.get_rang(sD, sD / 2, level, False))
            sD //= 2

        # обработка
        for i in range(self.max_tree_depth):
            # если разбиения на этом уровне нет, то закончить обработку
            if self.levelRange[i] == []:
                break
            else:
                self.maxFract = i

            for rang in self.levelRange[i]:
                findDomain = False
                # погрешность
                bestDomain = 0
                minEps = float('inf')
                # сначала изменяем домен, потом его сжимаем, потом проверяем на соответствие
                for domain in self.domain_list:
                    if domain.size > rang.size:
                        # рассмотрим все преобразования для домена
                        for direction, angle in self.candidates:
                            convDomain = self.apply_transformation(domain.data, direction, angle)
                            convDomain = self.rotate(convDomain, domain.size // rang.size)  # сжатый измененный домен
                            # найдем контраст и яркость
                            s, o = self.brightness_contrast(rang, convDomain)
                            convDomain = convDomain * s + o

                            # высчитываем разницу попиксельно
                            epsD = np.sum(np.square(rang, convDomain))
                            if epsD < minEps:
                                minEps = epsD
                                bestDomain = (domain.x, domain.y, domain.size, direction, angle, s, o)
                        if minEps < self.eps:
                            rang.domainSource = bestDomain
                            findDomain = True
                            break
                if findDomain is False:
                    if self.rang.level < self.max_tree_depth:
                        #разбиваем ранг на след уровень
                        nextLevelTree = rang.get_next_level_tree()

                        newSize = rang.size/2
                        newCoord = [(rang.x, rang.y), (rang.x, rang.y+newSize), (rang.x+newSize, rang.y+newSize), (rang.x+newSize, rang.y)]
                        for i in range(4):
                            self.levelRange[rang.level+1].append(range_struct(size=newSize, step=newSize, x=newCoord[i][0], y=newCoord[i][1], img=self.img, level=rang.level+1))
                    else:
                        rang.domainSource = bestDomain











