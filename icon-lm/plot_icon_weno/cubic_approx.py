from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


# Define the cubic function to fit
def cubic_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def fix_x_in_range(xmin, xmax, function):
    # Generate x values in the range [-1, 1]
    x_values = np.linspace(xmin, xmax, 100)
    # Compute y values for these x values using the function
    y_values = function(x_values)
    # Fit the cubic function to the original function
    parameters, _ = curve_fit(cubic_function, x_values, y_values)
    return parameters

# a, b, c, d = -0.5, 1.5, 1.2, 0.8
a, b, c, d = 1, -1, 1, 1
original_function = lambda x: a * np.sin(c * x) + b * np.cos(d * x)
original_range = [-1,2]

'''
apply change of variable, new_x = (x - xmin) / (xmax - xmin) * (new_xmax - new_xmin) + new_xmin
so that new_x is in the range [new_xmin, new_xmax]
x = (new_x - new_xmin) / (new_xmax - new_xmin) * (xmax - xmin) + xmin
then new_f(new_x) = f(x) = f((new_x - new_xmin) / (new_xmax - new_xmin) * (xmax - xmin) + xmin)
'''
new_range_list = [[-1,1], [-1,3]]
plt.figure(figsize=(4*2, 3))
i = 0
for new_range in new_range_list:
    i+=1
    plt.subplot(1,2,i)
    xmin, xmax = original_range
    new_xmin, new_xmax = new_range
    new_function = lambda new_x: original_function((new_x - new_xmin) / (new_xmax - new_xmin) * (xmax - xmin) + xmin)

    x_plot = np.linspace(new_xmin, new_xmax, 100)
    plt.plot(x_plot, new_function(x_plot), color='black')
    params = fix_x_in_range(new_xmin, new_xmax, new_function)
    plt.plot(x_plot, cubic_function(x_plot, *params), label='{:.3f}x^3 + {:.3f}x^2 + {:.3f}x + {:.3f}'.format(*params), color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Approximation in the range [{}, {}]'.format(new_xmin, new_xmax))
    plt.legend()
    plt.grid(True)

plt.savefig(f'cubic_approx.pdf')