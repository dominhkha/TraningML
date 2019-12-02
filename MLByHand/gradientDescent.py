

# Simple Gradient Descent 

# dominhkha
# 2/12/19


import numpy  as np 
import random


class Gradient:
    # def __init__(self):
        
    
    def square(self,x):
        return x**2
    
    def derivative(self,x):
        return x*2
    
    def sum_of_square(self,v):
        return sum(self.square(v_i) for v_i in v)

    def sum_of_square_gradient(self,v):
        return [self.derivative(v_i) for v_i in v]
    
    def step(self,direction,step_size,v):
        return [v_i + step_size* direction_i for v_i,direction_i in zip(v,direction)]

    def distance_between_two_vectors(self,v1,v2):
        return sum((v1_i-v2_i)**2 for v1_i,v2_i in zip(v1,v2))

    def quadaric(self, x):
        return [x_i**2+2*x_i+1 for x_i in x]
    def quadaric_derivative(self,x):
        return [2*x_i+2 for x_i in x]
    
            

if __name__=="__main__":

    tolerance = 0.00000000000000000001
    # v = [random.randint(-10,10) for i in range(3)]
    model = Gradient()
    v=[10]
    while True:
        direction = model.quadaric_derivative(v)
        v_next = model.step(direction,-0.01,v)
        print(v_next)
        if model.distance_between_two_vectors(v,v_next) < tolerance:
            break
        v=v_next
    
    print(v)
