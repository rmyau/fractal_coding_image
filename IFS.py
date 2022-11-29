import numpy as np
from scipy import ndimage
import matplotlib.image as mpigm
import matplotlib.pyplot as plt
from rangTree import *
import math


class cIFS:


    def __init__(self, file_name='', max_tree_depth=1, eps=0.0, maxLevelDomain=0):
        self.file_name = file_name
        # уровень макс достигнутый в дереве
        self.max_fract = 0
        # погрешность
        self.eps = eps
        self.domain_list = []
        # размерность блоков рангов на нулевом уровне
        self.max_tree_depth = max_tree_depth - 1
        self.maxlevelDomain = maxLevelDomain
        # значения афинных преобразований
        directions = [1, -1]  # кооэфициент для отражения
        angles = [0, 90, 180, 270]  # углы для афинных преобразований
        self.candidates = [[direction, angle] for direction in directions for angle in angles]
        if maxLevelDomain == 0:
            self.decompress()
        else:
            self.compress()
        self.startSizeRange = None
        self.height = None
        self.width = None

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

    def apply_transformation(self, img, direction, angle, contrast=1.0, brightness = 0.0):
        return contrast*self.rotate(self.flip(img, direction), angle)+brightness

    @staticmethod
    def reduce(img, factor):  # img - в данном случае для домена это data
        result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])

        return result

    @staticmethod
    def brightness_contrast(rang, domain):
        size = rang.size
        print(rang, domain)
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

    def find_contrast_and_brightness2(self, rang, domain):
        # Fit the contrast and the brightness
        A = np.concatenate((np.ones((domain.size, 1)), np.reshape(domain, (domain.size, 1))), axis=1)
        b = np.reshape(rang, (rang.size,))
        x, _, _, _ = np.linalg.lstsq(A, b)
        # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
        return x[1], x[0]
        # создание вектора доменов и рангов
    def get_rang(self, size, step, level, isRang):  # int,int,bool
        w = self.width
        h = self.height
        if not isRang:
            h -= step
            w -= step
        vector = []
        indexRang = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                indexRang += 1
                data = self.img[y:y + size, x:x + size]
                # блок - экземпляр домена - хранить в списке доменов
                block = range_struct(size=size, step=step, x=x, y=y, level=level, data=data)
                if isRang:
                    block.arr = byte(self.max_tree_depth).arr
                    block.arr[0] = indexRang
                vector.append(block)
        return vector

    def get_all_domain(self):
        sD = self.startSizeRange
        for level in range(self.maxlevelDomain - 1):
            self.domain_list.extend(self.get_rang(sD, sD // 2, level, False))
            sD //= 2

    def new_level_rang(self, rang):
        nextLevelTree = rang.get_next_level_tree()
        newSize = rang.size // 2
        newCoord = [(rang.x, rang.y), (rang.x, rang.y + newSize), (rang.x + newSize, rang.y),
                    (rang.x + newSize, rang.y + newSize)]
        newLev = rang.level + 1
        for j in range(4):
            newX = newCoord[j][0]
            newY = newCoord[j][1]
            data = self.img[newY:newY + newSize, newX:newX + newSize]
            new_rang: range_struct = range_struct(size=newSize, step=newSize, x=newX, y=newY, level=newLev, data=data)
            new_rang.arr = nextLevelTree[j]
            self.levelRange[newLev].append(new_rang)

    def find_best_transformation(self, rang, domain, minEps, bestDomain):
        print('NEXT DOMAIN--')
        print(f'in best tr x={domain.x} y={domain.y}')
        for direction, angle in self.candidates:
            convDomain = self.apply_transformation(domain.data, direction, angle)
            convDomain = self.reduce(convDomain, domain.size // rang.size)  # сжатый измененный домен
            # найдем контраст и яркость
            # s, o = self.brightness_contrast(rang, convDomain)
            s,o = self.find_contrast_and_brightness2(rang.data, convDomain)
            convDomain = convDomain * s + o
            # высчитываем разницу попиксельно

            epsD = np.sum(np.square(rang.data - convDomain)) / (rang.size * rang.size)
            if epsD < minEps:
                minEps = float(epsD)
                print(f'coord x={domain.x} y={domain.y}')
                bestDomain = (domain.x, domain.y, domain.size, direction, angle, s, o)
        return minEps, bestDomain

    def find_domain_for_rang(self, rang):
        findDomain = False
        # погрешность
        bestDomain = 0
        minEps = 2.0
        # сначала изменяем домен, потом его сжимаем, потом проверяем на соответствие
        for ind in range(len(self.domain_list)):
            domain = self.domain_list[ind]
            if domain.size > rang.size:
                # рассмотрим все преобразования для домена
                # minEps, bestDomain = self.find_best_transformation(rang, domain, minEps, bestDomain)
                minEps, bestDomain = self.find_best_transformation(rang, domain, minEps, bestDomain)

                if minEps < self.eps:
                    rang.domainSource = bestDomain
                    findDomain = True
        if findDomain is False:
            if rang.level < self.max_tree_depth - 1:
                self.new_level_rang(rang)
            else:
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

        self.get_all_domain()
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

    def find_size_vector(self):
        vector_size = []
        s = self.startSizeRange
        for i in range(self.max_tree_depth):
            vector_size.append(s)
            s //= 2
        return vector_size

    def find_coordinates_rang(self, index_rang_arr):
        x, y = 0, 0
        size_level = self.startSizeRange
        for ind in index_rang_arr:
            if ind == 0:
                break
            if ind == 1 or ind == 2:
                x += size_level
            if ind == 2 or ind == 4:
                y += size_level
            size_level //= 2
        return [x, y, size_level*2]

    def decompress(self):
        info_rangs = self.read_file()
        old_image = [np.random.randint(0, 256, (self.height, self.width))]
        size_level_rang = self.find_size_vector()
        for iter_ in range(6):
            new_image = np.zeros((self.width, self.height))
            print(iter_)
            #[xr,yr,sr], xd, yd, sd, direction, angle, s, 0
            for info in info_rangs:
                rang_x, rang_y, rang_size = int(info[0][0]), int(info[0][1]), int(info[0][2])
                # print(rang_x, rang_y, rang_size)
                domain_x, domain_y, domain_size = int(info[1]), int(info[2]), int(info[3])
                direction, angle = int(info[4]), int(info[5])
                s, o = float(info[6]), float(info[7])
                domain_data = old_image[-1][domain_y:domain_y+domain_size, domain_x:domain_x+domain_size]
                convDomain = self.apply_transformation(domain_data, direction, angle, s, o)
                convDomain = self.reduce(convDomain, domain_size // rang_size)

                new_image[rang_y:rang_y+rang_size, rang_x:rang_x+rang_size] = convDomain
            print(new_image)
            old_image.append(new_image)
        self.plot_iterations(old_image)
        plt.show()

    def plot_iterations(self, iterations, target=None):
        # Configure plot
        plt.figure()
        nb_row = math.ceil(np.sqrt(len(iterations)))
        nb_cols = nb_row
        # Plot
        for i, img in enumerate(iterations):
            plt.subplot(nb_row, nb_cols, i + 1)
            plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
            if target is None:
                plt.title(str(i))
            else:
                # Display the RMSE
                plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
        plt.tight_layout()



    def show_image(self, img):
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.show()
    def read_file(self):
        file = open(self.file_name, 'rb')
        size = [int(x) for x in file.readline().split()]
        self.width, self.height = size[0], size[1]
        self.startSizeRange = self.width // 2
        info_file = [x.split() for x in file.readlines()]
        file.close()
        self.max_tree_depth = len(info_file[0][0])
        for i in range(len(info_file)):
            arr_rang = [int(x) for x in list(str(int(info_file[i][0])))]
            info_file[i][0] = self.find_coordinates_rang(arr_rang)

        return info_file


    @staticmethod
    def checkImage(img, startSizeRange):
        if img.get_width() % startSizeRange != 0:
            return False
        if img.get_height() % startSizeRange != 0:
            return False
        return True


# test_ifc = cIFS("monkey.gif", 5, 0.4, 4)
# test_ifc = cIFS("monkey.png", 5, 0.005, 4)
# print(f'start size {test_ifc.startSizeRange}')
decode_ifc = cIFS(file_name='monkey.fbr')
# test_ifc.compress()
