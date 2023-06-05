import numpy as np

class Functions:
    Tensor = None

    def _convert_to_tensor(self, other):
        if not isinstance(self, Functions.Tensor):
            self = Functions.Tensor(self)
        if other and not isinstance(other, Functions.Tensor):
            other = Functions.Tensor(other)
        
        return self, other
    
    def unary_operation(self, func):
        self, _ = Functions._convert_to_tensor(self, None)
        return func(self)
    
    def binary_operation(self, other, func):
        self, other = Functions._convert_to_tensor(self, other)
        return func(self, other)
    
    def _add(self, other):
        res = Functions.Tensor(self.data + other.data, children=(self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += res.grad
                other.grad += res.grad

            res._backward = _backward

        return res
    
    def _mul(self, other):
        res = Functions.Tensor(self.data * other.data, children=(self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += res.grad * other.data
                other.grad += res.grad * self.data

            res._backward = _backward

        return res
    
    def _sub(self, other):
        res = Functions.Tensor(self.data - other.data, children=(self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += res.grad
                other.grad += - res.grad

            res._backward = _backward

        return res
    
    def _truediv(self, other):
        assert other.data != 0, "Division by 0 error"

        res = Functions.Tensor(self.data / other.data, children=(self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += res.grad / other.data
                other.grad += - res.grad * self.data / other.data ** 2

            res._backward = _backward

        return res
    
    def _pow(self, other):
        res = Functions.Tensor(self.data ** other.data, children = (self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += res.grad * other.data * self.data ** (other.data - 1)
                other.grad += res.grad * self.data ** other.data * np.log(self.data)

            res._backward = _backward

        return res
    
    def _matmul(self, other):
        res = Functions.Tensor(np.matmul(self.data, other.data), children = (self, other))

        if self.requires_grad or other.requires_grad:
            def _backward():
                self.grad += np.matmul(res.grad, other.data.T)
                other.grad += np.matmul(self.data.T, res.grad)

            res._backward = _backward

        return res
    
    def _neg(self):
        res = Functions.Tensor(-self.data, children=(self, ))

        if self.requires_grad:
            def _backward():
                self.grad += - res.grad

            res._backward = _backward

        return res
    
    def _T(self):
        res = Functions.Tensor(np.transpose(self.data), children=(self, ))

        if self.requires_grad:
            def _backward():
                self.grad += np.transpose(self.grad)

            res._backward = _backward

        return res
    
    def _min(self):
        res = Functions.Tensor(np.min(self.data), children=(self, ))

        if self.requires_grad:
            def _backward():
                self.grad += res.grad * (self.data == res.data)

            res._backward = _backward

        return res
    
    def _max(self):
        res = Functions.Tensor(np.max(self.data), children=(self, ))

        if self.requires_grad:
            def _backward():
                self.grad += res.grad * (self.data == res.data)

            res._backward = _backward

        return res
    
    def _mean(self):
        res = Functions.Tensor(np.mean(self.data), children=(self, ))

        if self.requires_grad:
            def _backward():
                self.grad += res.grad / np.size(self.data)

            res._backward = _backward

        return res

    def add(self, other):
        return Functions.binary_operation(self, other, Functions._add)
    
    def mul(self, other):
        return Functions.binary_operation(self, other, Functions._mul)
    
    def sub(self, other):
        return Functions.binary_operation(self, other, Functions._sub)
    
    def truediv(self, other):
        return Functions.binary_operation(self, other, Functions._truediv)
    
    def pow(self, other):
        return Functions.binary_operation(self, other, Functions._pow)
    
    def matmul(self, other):
        return Functions.binary_operation(self, other, Functions._pow)
    
    def neg(self):
        return Functions.unary_operation(self, Functions._neg)
    
    def T(self):
        return Functions.unary_operation(self, Functions._T)
    
    def min(self):
        return Functions.unary_operation(self, Functions._min)
    
    def max(self):
        return Functions.unary_operation(self, Functions._max)
    
    def mean(self):
        return Functions.unary_operation(self, Functions._mean)
