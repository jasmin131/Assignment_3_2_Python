# Assignment 1.2P Solutions - 2024 Programming in Psychological Science
#
# Date            Programmer              Descriptions of Change
# ====         ================           ======================
# 19-Jan-24     Jasmin Hagemann               Original code

import numpy as np
from datetime import datetime
import time
import warnings

# Q2.2P.1 ----------------------------------------------------------------------

from datetime import datetime, time

time_now = datetime.now().time()

if time(0, 0) <= time_now <= time(5, 0):
    print ("Go to sleep!")
elif time(7, 0) <= time_now <= time(10, 0):
    print("Eet je hagelslag!")
else:
    print ("Gut gemacht!")

# Q2.2P.2 ----------------------------------------------------------------------

numeric_vec = np.array([1, 2, 1, 4])

weighted_sum = 0

for i in range(len(numeric_vec)):

    if i % 2 == 0:
        new_element = numeric_vec[i] * 2
    else:
        new_element = numeric_vec[i]

    weighted_sum += new_element

weighted_average = weighted_sum / (np.size(numeric_vec)*1.5)

# Q2.2P.3 ----------------------------------------------------------------------

grass = "green"

def color_it(color_me, grass_me):
    grass_me = grass
    color_me = "blue"
    # grass = "blue"
    colorful_items = np.array([(color_me, grass_me)])
    return colorful_items

sky = "grey"
ground = "brown"
these_items = color_it(sky, ground)
print(these_items)

# a) Contrary to R, this code does not run in Python as an error message arises:
# UnboundLocalError: cannot access local variable 'grass' where it is not associated with a value
# The difference between R and Python is that Python has a strict separation between global and local scopes,
# while R searches for variables in the global scope if they are not found in the local scope.
# Moreover, Python requires explicit declaration to modify global variables within a function,
# whereas R creates a local copy unless it is instructed to use a global variable.

# Q2.2P.4 ----------------------------------------------------------------------

import numpy as np
def unique_values(vec):
    """this function calculates all unique values of a Numpy vector"""

    if not isinstance(vec, np.ndarray):
        raise Exception ("Input must be an array!")

    unique_values_vec = np.array([])

    for element in vec:
        # unique values of a Numpy vector
        if element not in unique_values_vec:
        # if sum(vec == element) == 1:
            unique_values_vec += element

    if np.array_equal(vec, unique_values_vec):
        raise Warning ("All values are special!")

    return unique_values_vec

unique_values(np.array([1, 2, 4, 5, 2, 7, 4]))

# Q2.2P.5 ----------------------------------------------------------------------

# a) The try block lets one test a block of code for errors.
# try/except blocks allow to try some code, and if an exception is raised (or 'thrown'),
# Then it is possible to catch it and execute different code.
# An exception that is caught will not cause the program to crash.

# b) Examples of try block usage

try:
    unique_values("hello")
except:
    ValueError ("Please enter arrays/vectors only")

try:
    unique_values(6)
except:
    ValueError ("Please enter arrays/vectors only")

# Q2.2P.6 ----------------------------------------------------------------------

class MyClass:
    """A simple example class"""
    classnum = 12345
    def famous(self):
        return 'hello world'

new_stuff = MyClass()
new_stuff.classnum
new_stuff.famous()

# A class is a programmer-defined type and is basically a blueprint/template for a type of object.
# It specifies what attributes the objects have and what methods can operate on them.
# When defining a class a new object type with the same name is created.

# Complex number class

import numpy as np
class ComplexNum:
    """Creates a complex number"""
    numtype = 'complex'

    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def vec_length(self):
        return np.sqrt(self.r**2 + self.i**2)

    def phase_angle(self):
        """ Function returns the phase angle in degrees of the complex vector"""
        # np.angle(my_num, deg=True)
        return np.rad2deg(np.arctan(self.i / self.r))

my_num = ComplexNum(3.0, 4.0)
print(my_num) # <__main__.ComplexNum object at 0x100cc0fd0>
print((my_num.r, my_num.i)) # (3.0, 4.0)
print(my_num.numtype) # complex
print(my_num.vec_length()) # 5.0
print(my_num.phase_angle()) # 53.13010235415598

# Q2.2P.7 ----------------------------------------------------------------------

import math

# goal: number_to_nth = number**n

# helper functions f1 and f2
def f1(number, n):
    return math.log(number, n)
def f2(number, n):
    return (1/n) * math.log(number)

def nthpower(number, n, start = 1, max_iterations = 10000):

    for i in range(max_iterations):
        power_n =  start - f1(number, n)/f2(number, n)
        start = power_n

    return power_n

# Q2.2P.8 ----------------------------------------------------------------------

def prime(n):
    """ The function returns the first n prime numbers as a numpy array
    Enter a number n to calculate n primes."""

    def is_prime(num):
        """ returns a boolean indicating whether a number is a prime or not"""
        if num <= 1:
            return False
        for i in range(2, num):
            if num % i == 0:
                return False
        return True

    primes = []
    num = 2

    while len(primes) < n :

        if is_prime(num):
            primes.append(num)
        num += 1
    return np.array(primes)

# Q2.2P.9 ----------------------------------------------------------------------

# into Python console:
# from sequences import prime
# ?prime

# when typing ?prime into my Python console I see my docstring of the prime function.

# for further answers please refer to sequences.py

# Q2.2P.10 ----------------------------------------------------------------------

# please refer to sequences.py

# Q2.2P.11 ----------------------------------------------------------------------

# please refer to videopoker.py
