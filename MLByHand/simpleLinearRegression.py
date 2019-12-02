
# Simple Linear Regression
# dominhkha
# 2/12/19

import numpy as np 

class SimpleLinearRegression:
    
    def target_fn(self,x,theta):
        return x*theta[0]+theta[1]
    
    def gradient_fn(self,x,y,theta):
        a = 0.25*sum([(-y_i + self.target_fn(x_i,theta))*x_i for x_i,y_i in zip(x,y)])
        
        b = 0.25*sum([-y_i + self.target_fn(x_i,theta) for x_i,y_i in zip(x,y)])
        
        return [a,b]

    def sum_of_square(self,x,y,theta):
        return sum([(y_i - x_i * theta[0]-theta[1])**2 for x_i,y_i in zip(x,y)]) 

    def update(self,x,y,theta,learning_rate=0.1):
        gradient =  self.gradient_fn(x,y,theta)
        return [theta_i - learning_rate*gradient_i for theta_i,gradient_i in zip(theta,gradient)]

    def distance_between_two_vectors(self,v1,v2):
        return sum((v1_i-v2_i)**2 for v1_i,v2_i in zip(v1,v2))

    def batchGradientDescent(self,x,y,theta):
        # print(theta)
        while True:
            next_theta= self.update(x,y,theta)
            # print(self.sum_of_square(x,y,theta))
            if self.distance_between_two_vectors(theta,next_theta) < 0.0000001:
                break
            theta = next_theta
            print(theta)
            # break


if __name__=="__main__":
    x=[1,2,3,4]
    y=[10*x_i + 1 for x_i in x]
    theta = [10,10]
    model = SimpleLinearRegression()
    model.batchGradientDescent(x,y,theta)


