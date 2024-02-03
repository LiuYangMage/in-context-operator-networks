from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

name = '$f = sin(u) - cos(u)$'
def original_function(x):
    return np.sin(x) - np.cos(x)

# Define the cubic function to fit
def cubic_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def fix_x_in_range(xmin, xmax):
    # Generate x values in the range [-1, 1]
    x_values = np.linspace(xmin, xmax, 100)
    # Compute y values for these x values using the original function
    y_values = original_function(x_values)
    # Fit the cubic function to the original function
    parameters, _ = curve_fit(cubic_function, x_values, y_values)
    return parameters


# Plotting
plt.figure(figsize=(5, 4))
x_plot = np.linspace(-2, 2, 100)
plt.plot(x_plot, original_function(x_plot), label=name, color='black')

params = [-1/6, 0.5, 1, -1]
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = -1/6u^3 + 1/2u^2 + u - 1$'.format(*params), 
            color='green', linestyle='--')

xmin, xmax = -1,1
params = fix_x_in_range(xmin, xmax)
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = {:.3f}u^3 + {:.3f}u^2 + {:.3f}u {:.3f}$'.format(*params), 
            color='blue', linestyle='--')

xmin, xmax = -2,2
params = fix_x_in_range(xmin, xmax)
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = {:.3f}u^3 + {:.3f}u^2 + {:.3f}u {:.3f}$'.format(*params), 
            color='red', linestyle='--')

plt.xlabel('$u$')
plt.ylabel('$f$')
plt.vlines(1, -2, 2, linestyle=':', color='black')
plt.vlines(-1, -2, 2, linestyle=':', color='black')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.legend()
plt.savefig(f'cubic_approx_{name[1:-1].replace(" ","_")}.pdf')



name = '$f = tanh(u)$'
def original_function(x):
    return np.tanh(x)


# Define the cubic function to fit
def cubic_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def fix_x_in_range(xmin, xmax):
    # Generate x values in the range [-1, 1]
    x_values = np.linspace(xmin, xmax, 100)
    # Compute y values for these x values using the original function
    y_values = original_function(x_values)
    # Fit the cubic function to the original function
    parameters, _ = curve_fit(cubic_function, x_values, y_values)
    return parameters


# Plotting
plt.figure(figsize=(5, 4))
x_plot = np.linspace(-2, 2, 100)
plt.plot(x_plot, original_function(x_plot), label=name, color='black')

params = [-1/3, 0, 1, 0]
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = - 1/3u^3 + u$', 
            color='green', linestyle='--')

xmin, xmax = -1,1
params = fix_x_in_range(xmin, xmax)
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = {:.3f}u^3 + {:.3f}u$'.format(params[0], params[2]),
            color='blue', linestyle='--')

xmin, xmax = -2,2
params = fix_x_in_range(xmin, xmax)
plt.plot(x_plot, cubic_function(x_plot, *params), label='$f = {:.3f}u^3 + {:.3f}u$'.format(params[0], params[2]),
            color='red', linestyle='--')

plt.xlabel('$u$')
plt.ylabel('$f$')
plt.vlines(1, -2, 2, linestyle=':', color='black')
plt.vlines(-1, -2, 2, linestyle=':', color='black')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.legend()
plt.savefig(f'cubic_approx_{name[1:-1].replace(" ","_")}.pdf')