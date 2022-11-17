






class conversationSet:
    def __init__(self):
        list = [self.withoutConv, self.turnY, ]

    def withoutConv(self,x,y,n):
        return x,y
    def turnY(self,x,y,n):
        #a[x,y] = a[x,n-1-y]
        return x,-y+n-1
    def turnX(self,x,y,n):
        #a[x,y]=a[n-1-x,y]
        return -x+n-1,y
    def turnMainDiag(self,x,y,n):
        return y,x
    def turnSecDiag(self,x,y,n):
        #a[x,y] = a[n-1-y,n-1-x]
        return -x+n-1,-y+n-1
    def turnLeft(self,x,y,n):
        #rotate -90, a[y,n-1-x]
        return y,-x+n-1
    def turnFull(self,x,y,n):
        #a[n-1-x,n-1-y]
        return -x+n-1, -y+n-1
    def turnRight(self,x,y,n):
        #a[n-1-y,x]
        return n-1-y,x




