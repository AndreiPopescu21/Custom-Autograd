import numpy as np
from collections import deque
from Functions import Functions as F

class Tensor:
    def __init__(self, data, children=(), requires_grad=True):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = 0
        self._backward = lambda: None
        self.children = children

        F.Tensor = self.__class__

    def backward(self):
        self.grad = 1
        q = deque([self])

        while len(q) > 0:
            current = q.pop()
            current._backward()

            for child in current.children:
                q.appendleft(child)

    def zero_grad(self):
        q = deque([self])

        while len(q) > 0:
            current = q.pop()
            current.grad = 0
            current._backward = lambda: None

            for child in current.children:
                q.appendleft(child)

    def __add__(self, other):
        return F.add(self, other)
    
    def __radd__(self, other):
        return F.add(other, self)
    
    # def __iadd__(self, other):
    #     return F.add(self, other)
    
    def __mul__(self, other):
        return F.mul(self, other)
    
    def __rmul__(self, other):
        return F.mul(other, self)
    
    # def __imul__(self, other):
    #     return F.mul(self, other)
    
    def __sub__(self, other):
        return F.sub(self, other)
    
    def __rsub__(self, other):
        return F.sub(other, self)
    
    # def __isub__(self, other):
    #     return F.sub(self, other)
    
    def __truediv__(self, other):
        return F.truediv(self, other)
    
    def __rtruediv__(self, other):
        return F.truediv(other, self)
    
    # def __itruediv__(self, other):
    #     return F.truediv(self, other)
    
    def __pow__(self, other):
        return F.pow(self, other)
    
    def __rpow__(self, other):
        return F.pow(other, self)
    
    # def __ipow__(self, other):
    #     return F.pow(self, other)
    
    def __matmul__(self, other):
        return F.matmul(self, other)
    
    def __rmatmul__(self, other):
        return F.matmul(other, self)
    
    # def __imatmul__(self, other):
    #     return F.matmul(self, other)
    
    def __neg__(self):
        return F.neg(self)
    
    def T(self):
        return F.T(self)
    
    def min(self):
        return F.min(self)
    
    def max(self):
        return F.max(self)
    
    def mean(self):
        return F.mean(self)
    
    def shape(self):
        return np.shape(self.data)
    
    def relu(self):
        pass

    def tanh(self):
        pass

    def sigmoid(self):
        pass

    def softmax(self):
        pass
    
    def __setitem__(self, item, data):
        self.data[item] = data

    def __getitem__(self, item):
        return self.data[item]
    
    def __repr__(self):
        return f'Tensor({self.data}, {self.grad})'
    
if __name__ == '__main__':
    pass