import numpy as np
from scipy import ndimage
import matplotlib.image as mpigm
import matplotlib.pyplot as plt
from rangTree import *


class cIFS:
    def __init__(self, file_name='', max_tree_depth=1, eps=0.0, maxLevelDomain=0):
        # уровень макс достигнутый в дереве
        self.max_fract = 0
        # погрешность
        self.eps = eps
        self.file_name = file_name
        # размерность блоков рангов на нулевом уровне
        self.max_tree_depth = max_tree_depth
        self.maxlevelDomain = maxLevelDomain
        # значения афинных преобразований
        self.directions = [1, -1]  # кооэфициент для отражения
        self.angles = [0, 90, 180, 270]  # углы для афинных преобразований
        self.candidates = [[direction, angle] for direction in self.directions for angle in self.angles]

    @staticmethod
    def rotate(img, angle):
        return ndimage.rotate(img, angle, reshape=False)

    @staticmethod
    def get_greyscale_image(img):
        # mean - average for massive
        return np.mean(img[:, :, :2], 2)

    @staticmethod
    def flip(img, direction):
        return img[::direction, :]

    def apply_transformation(self, img, direction, angle):
        return self.rotate(self.flip(img, direction), angle)

    def reduce(self, img, factor):  # img - в данном случае для домена это data
        # двумерный массив, заполненный нулями, соответствует размерности изображения
        # img.shape[0] - количество пикселей

        result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])

        return result

    @staticmethod
    def brightness_contrast(rang, domain):
        size = rang.size
        dataD = np.array(domain)
        dataR = np.array(rang.data)
        sumD = np.sum(dataD)
        sumR = np.sum(dataR)

        d = (1 / (size * size)) * sumD
        r = (1 / (size * size)) * sumR
        b = np.sum(np.square(dataD - d))

        newD = dataD - d
        newR = dataR - r
        res = np.array([[0.0 for i in range(size)] for j in range(size)])
        for i in range(size):
            for j in range(size):
                res[i][j] = newD[i][j] * newR[i][j]
        a = np.sum(res)
        o = r - (a / b) * d
        s = a / b
        return s, o

    # создание вектора доменов и рангов
    def get_rang(self, size, step, level, isRang):  # int,int,bool
        w = self.width
        h = self.height
        if not isRang:
            h -= step
            w -= step
        vector = []
        indexRang = 0
        print(f'step = {step}, h={h}, w={w}, size={size}')
        for y in range(0, h, step):
            for x in range(0, w, step):
                indexRang += 1
                data = self.img[y:y + size, x:x + size]
                # блок - экземпляр домена - хранить в списке доменов
                block = range_struct(size=size, step=step, x=x, y=y, level=level, data=data)
                if isRang:
                    block.arr = byte(self.max_tree_depth).arr
                    block.arr[0] = indexRang
                    print(f'block arr {block.arr}')
                vector.append(block)
        return vector

    def new_level_rang(self, rang):
        nextLevelTree = rang.get_next_level_tree()
        newSize = rang.size // 2
        newCoord = [(rang.x, rang.y), (rang.x, rang.y + newSize), (rang.x + newSize, rang.y),
                    (rang.x + newSize, rang.y + newSize)]
        print(newCoord)
        newLev = rang.level + 1
        for j in range(4):
            newX = newCoord[j][0]
            newY = newCoord[j][1]
            data = self.img[newY:newY + newSize, newX:newX + newSize]
            new_rang: range_struct = range_struct(size=newSize, step=newSize, x=newX, y=newY, level=newLev, data=data)
            new_rang.arr = nextLevelTree[j]
            self.levelRange[newLev].append(new_rang)

    def find_best_transformation(self, rang, domain, minEps, bestDomain):
        for direction, angle in self.candidates:
            convDomain = self.apply_transformation(domain.data, direction, angle)
            convDomain = self.reduce(convDomain, domain.size // rang.size)  # сжатый измененный домен
            # найдем контраст и яркость
            s, o = self.brightness_contrast(rang, convDomain)
            convDomain = convDomain * s + o
            # высчитываем разницу попиксельно
            epsD = np.sum(np.square(rang.data, convDomain)) / (rang.size * rang.size)
            if epsD < minEps:
                minEps = epsD
                bestDomain = (domain.x, domain.y, domain.size, direction, angle, s, o)
        return minEps, bestDomain

    def get_all_domain(self):
        domain_list = []
        sD = self.startSizeRange
        for level in range(self.maxlevelDomain - 1):
            domain_list.extend(self.get_rang(sD, sD // 2, level, False))
            sD //= 2
        return domain_list

    def find_domain_for_rang(self, rang):
        findDomain = False
        # погрешность
        bestDomain = 0
        minEps = float('inf')
        # сначала изменяем домен, потом его сжимаем, потом проверяем на соответствие
        for domain in self.domain_list:
            if domain.size > rang.size:
                # рассмотрим все преобразования для домена
                minEps, bestDomain = self.find_best_transformation(rang, domain, minEps, bestDomain)

                if minEps < self.eps:
                    rang.domainSource = bestDomain
                    findDomain = True
                    print(f'find domain eps -- {minEps}')
                    break
        print(f'eps  ----  {minEps}')
        if findDomain is False:
            print('Разбиваем на новый уровень')
            if rang.level < self.max_tree_depth - 1:
                self.new_level_rang(rang)
            else:
                print('find domain')
                rang.domainSource = bestDomain

    def compress(self):  # eps - погрешность
        self.img = mpigm.imread(self.file_name)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.startSizeRange = self.width // 2
        # checkImage
        s = self.startSizeRange

        self.levelRange = [[] for i in range(self.max_tree_depth)]

        # найдем начальный уровень рангов
        self.levelRange[0] = self.get_rang(s, s, 0, True)
        for rang in self.levelRange[0]:
            self.new_level_rang(rang)

        domain_list = self.get_all_domain()
        # обработка
        for i in range(1, self.max_tree_depth):
            # если разбиения на этом уровне нет, то закончить обработку
            if self.levelRange[i] == []:
                break
            else:
                self.max_fract = i
            for rang in self.levelRange[i]:
                self.find_domain_for_rang(rang)

        self.save_file()

    @staticmethod
    def get_info_rang(rang):
        info = str(''.join(str(x) for x in rang.arr)) + ' '
        info += f'{rang.domainSource[0]} {rang.domainSource[1]} {rang.domainSource[2]} '
        info += f'{rang.domainSource[3]} {rang.domainSource[4]} {rang.domainSource[5]} {rang.domainSource[6]}\n'
        return info

    def save_file(self):
        output = open(f'{self.file_name[:-4]}.fbr', 'wb')
        output.write(f'{self.width} {self.height}\n'.encode())
        for level in self.levelRange:
            for rang in level:
                if not rang.haveNextLevel:
                    output.write(self.get_info_rang(rang).encode())
        output.close()


    @staticmethod
    def checkImage(img, startSizeRange):
        if img.get_width() % startSizeRange != 0:
            return False
        if img.get_height() % startSizeRange != 0:
            return False
        return True


# test_ifc = cIFS("monkey.gif", 5, 0.4, 4)
test_ifc = cIFS("monkey_256_grey.png", 5, 0.4, 4)
print(f'start size {test_ifc.startSizeRange}')

test_ifc.compress()
