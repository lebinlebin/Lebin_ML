
#Header
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Define a function(here a exponential function is used)
def func(x, a, b, c):
 return a * np.exp(-b * x) + c

#Create the data to be fit with some noise
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'bo', label='data')

#Fit for the parameters a, b, c of the function func:
popt, pcov = curve_fit(func, xdata, ydata)
popt #output: array([ 2.55423706, 1.35190947, 0.47450618])
plt.plot(xdata, func(xdata, *popt), 'r-',
 label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

#In the case of parameters a,b,c need be constrainted
#Constrain the optimization to the region of
#0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
popt #output: array([ 2.43708906, 1. , 0.35015434])
plt.plot(xdata, func(xdata, *popt), 'g--',
 label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

#Labels
plt.title("Exponential Function Fitting")
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
leg = plt.legend()  # remove the frame of Legend, personal choice
leg.get_frame().set_linewidth(0.0) # remove the frame of Legend, personal choice
#leg.get_frame().set_edgecolor('b') # change the color of Legend frame
#plt.show()

#Export figure
#plt.savefig('fit1.eps', format='eps', dpi=1000)
plt.savefig('fit1.pdf', format='pdf', dpi=1000, figsize=(8, 6), facecolor='w', edgecolor='k')
# plt.savefig('fit1.jpg', format='jpg', dpi=1000, figsize=(8, 6), facecolor='w', edgecolor='k')
