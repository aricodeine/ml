# import matplotlib.pyplot as plt

# # Sample data
# x = [1, 2, 3, 4, 5,6,7,8,6]
# y = [2, 4, 5, 4,3,5,6,8,7]

# # Calculate the means of x and y
# mean_x = sum(x) / len(x)
# mean_y = sum(y) / len(y)

# # Calculate the slope (m)
# numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
# denominator = sum((xi - mean_x) ** 2 for xi in x)
# m = numerator / denominator

# # Calculate the y-intercept (b)
# b = mean_y - m * mean_x

# # Create a function to make predictions
# def predict(x_value):
#     return m * x_value + b

# # Test the regression model with a new x value
# new_x = 6
# predicted_y = predict(new_x)
# print("Predicted y:", predicted_y)

# # Create a scatter plot of the data
# plt.scatter(x, y, label="Data Points")

# # Create the regression line using the calculated slope and intercept
# regression_line = [predict(xi) for xi in x]

# # Plot the regression line
# plt.plot(x, regression_line, label="Regression Line", color='red')

# # Add labels and a legend
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()

# # Display the plot
# plt.show()



import matplotlib.pyplot as plt


x=[1,2,3,4,5]
y=[2,4,5,4,5]


x_mean=sum(x)/len(x)
y_mean=sum(y)/len(y)


numerator=sum((xi-x_mean)*(yi-y_mean) for xi,yi in zip(x,y))
denominator=sum((xi-x_mean)**2 for xi in x)


m=numerator/denominator

b=y_mean-m*x_mean

def predict(x):
    return m*x+b


new_x=int(input("Enter the value of x for testing the regression model : "))

print(f'predicted value of y is: {predict(new_x)}')

plt.scatter(x,y,label='data points')

regression_line=[predict(val)for val in x]

plt.plot(x,regression_line,color='red',label='regression line')

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()

