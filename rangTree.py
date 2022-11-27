import numpy as np
# структура ранга
class range_struct:
    def __init__(self, x, y, size, step, level, data):
        #массив с номером в дереве для ранга
        self.arr = 0
        #self.domain_index = -1 - добавить в процессе обработки
        self.x=x
        self.y=y
        self.size=size
        self.step=step
        self.level = level
        #массив изображения в растре для заданного блока
        self.data = data
        #для рангов
        self.domainSource = 0 #хранит координаты доменного блока, размер, отражение, угол, контраст, яркость
        #если данный показатель истинный - ранг не храним в сжатом файле
        self.haveNextLevel = False

    def set_raster(self):
        data = [[] for j in range(self.size)]
        print(f'size in raster {self.size}, x= {self.x}, y = {self.y}')
        for i in range(self.size):
            for j in range(self.size):
                # print(f'i = {i},  j = {j}')
                data[i].append(self.img[self.x + i, self.y + j])
        return data

    def get_next_level_tree(self):
        self.haveNextLevel = True
        #добавить исключения
        if self.arr==0:
            print('Error in rang')
            return
        vector = []
        for i in range(1, 5):
            b = self.arr.copy()
            b[self.level+1] = i
            vector.append(b)
        print(f'vector {vector}')
        return vector


# структура одного уровня - число, которое хранится в виде массива, каждая цифра отвечает за уровень дерева
class byte:
    def __init__(self, depth):
        self.arr = [0 for i in range(depth)]


# levelArray = []
# # мощность уровня
# power_4 = [0 for i in range(max_tree_depth)]
# power_4[0] = 1
