import numpy as np
import matplotlib.pyplot as plt



def obj(x):
    return x**2


def gradient(x):
    return 2*x

initial_x=5
learning_rate=0.1
itr=20

x=initial_x
x_history=[x]
loss_history=[obj(x)]


for i in range(itr):
    grad=gradient(x)
    x-=learning_rate*grad
    
    x_history.append(x)
    loss_history.append(obj(x))
    print('Iteration ',i)
    print('x:',x)
    print('f(x):',obj(x))
    
x_values=np.linspace(-6,6,400)
y_values=obj(x_values)

plt.plot(x_values,y_values,label="Objective Function")
plt.scatter(x_history,loss_history,color='red',label="Gradient Descent Function ")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()


