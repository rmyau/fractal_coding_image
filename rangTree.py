import numpy as np
# структура ранга
class range_struct:
    def __init__(self,x,y,size,step,img, level):
        #массив с номером в дереве для ранга
        self.arr = 0
        #self.domain_index = -1 - добавить в процессе обработки
        self.img=img
        self.x=x
        self.y=y
        self.size=size
        self.step=step
        self.level = level
        #массив изображения в растре для заданного блока
        self.data=self.set_raster()
        #для рангов
        self.domainSource = 0 #хранит координаты блока, размер, отражение, угол, контраст, яркость
        self.haveNextLevel = False

    def set_raster(self):
        if self.data!=[]:
            self.data=[[0 for i in range(self.size)] for j in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.data[i][j] = self.img[self.x, self.y]

    def (self):
        self.haveNextLevel = True
        if self.arr==0:
            print('Error in rang')
            return
        vector = []
        for i in range(1, 5):
            b=self.arr
            b[self.level+1]=i
            vector.append(b)
        return vector


# структура одного уровня - число, которое хранится в виде массива, каждая цифра отвечает за уровень дерева
class byte:
    def __init__(self, depth):
        self.arr = [0 for i in range(depth)]


# levelArray = []
# # мощность уровня
# power_4 = [0 for i in range(max_tree_depth)]
# power_4[0] = 1
