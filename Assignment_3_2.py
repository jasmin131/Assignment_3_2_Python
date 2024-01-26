# Assignment 3.2P Solutions - 2024 Programming in Psychological Science
#
# Date            Programmer              Descriptions of Change
# ====         ================           ======================
# 26-Jan-24     Jasmin Hagemann               Original code

import sys
import pylint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Q3.2P.1 ----------------------------------------------------------------------

uni_data = np.random.uniform(-10, 10, size=1000)

plt.hist(uni_data)

# creates violin plot
sns.violinplot(uni_data, color="grey")
# adds jittered data points
sns.stripplot(uni_data, jitter=True)

plt.title("Q3.2P.1 - Violin plot with Jittered Data Points")

plt.savefig("violin_plot.png", dpi=300)

plt.close("all")

# I find the violin plot more insight full.

# Q3.2P.2 ----------------------------------------------------------------------

titanic_data = pd.read_csv("https://raw.githubusercontent.com/hannesrosenbusch/schiphol_class/master/titanic.csv")

plt.figure()
sns.violinplot(data=titanic_data, x='Survived', y='Age')

plt.title("Distribution of Age for Survivors and Non-Survivors on the Titanic")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.savefig("titanic_data.png", dpi=300)

plt.close("all")

# The graph does not show any remarkable differences between the passengers who survived compared to those who died.

# Q3.2P.3 ----------------------------------------------------------------------

# tips dataset from the seaborn package
tips = sns.load_dataset('tips')

# scatter plot with total_bill on the x axis and tip on the y axis
plt.scatter(tips["total_bill"], tips["tip"])

# Add regression line
sns.regplot(x='total_bill', y='tip', data=tips, scatter=False, color='orange')

plt.xlabel('Total Bill', fontsize=16)
plt.ylabel('Tip', fontsize=16)

plt.title("Q3.2P.3 - Scatter Plot of Total Bill vs. Tip")

plt.savefig("scatterplot_tips.png", dpi=300)

plt.close("all")

# Q3.2P.4 ----------------------------------------------------------------------

# Plot a heatmap showing all the correlations between the numeric variables in the diamonds dataset from seaborn.
# Next to the heatmap, add a kdeplot with carat on the x axis and price on the y axis.

diamonds = sns.load_dataset('diamonds')

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
diamonds_num = diamonds.select_dtypes(include=numerics)
corr_matrix = diamonds_num.corr()

plt.figure()

plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix)
plt.title("Correlation Heatmap of Diamonds Dataset", fontsize=8)

plt.subplot(1, 2, 2)
sns.kdeplot(x="carat", y="price", data=diamonds)
plt.title("KDE Plot of Carat vs. Price", fontsize=8)
plt.xlabel("Carat")
plt.ylabel("Price")

plt.tight_layout()
plt.subplots_adjust(wspace=0.5)

plt.savefig("subplots_heatmap_kdeplot.png", dpi=300)

plt.close("all")

# Q3.2P.5 ----------------------------------------------------------------------

import random
import matplotlib.pyplot as plt
import numpy as np

def my_plots (input_string):
    if input_string == "eww":
        # confusing, non-informative plot
        plt.figure()
        random_data = np.random.rand(10)
        random_dependent = np.random.rand(10)
        plt.plot(random_data, random_dependent)
        plt.scatter(random_data, random_dependent)
        plt.xlabel("y-axis")
        plt.ylabel("x-axis")
    elif input_string == "yay":
        # clear, informative plot
        plt.figure()
        titanic_data = pd.read_csv("https://raw.githubusercontent.com/hannesrosenbusch/schiphol_class/master/titanic.csv")
        plt.bar(titanic_data["Sex"], titanic_data["Age"])
        plt.xlabel("Gender of Passengers")
        plt.ylabel("Age of Passengers")
    else:
        print("Incorrect Argument: The function can either be used with “eww” or “yay”!",
              "\n", "Please run function again with correct argument.")

# Q3.2P.6 ----------------------------------------------------------------------

# Old bad code

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


# New good code

def is_prime(num):

    """ returns a boolean indicating whether a number is a prime or not """

    if num <= 1:
        return False
    for i in range(2, num):
        if num % i == 0: # if modulo of the iterating number is zero, number is dividable (not a prime)
            return False
    return True

def prime(n):

    """ The function returns the first n prime numbers as a numpy array
    Enter a number n to calculate n primes.
    """

    primes = []
    num = 2

    while len(primes) < n : # loop runs until the indicated number of primes is generated
        if is_prime(num):
            primes.append(num)
        num += 1
    return np.array(primes)

# Changes:
# 1) I put the first function outside of the second function.
# 2) I changed the quotations of my docstring. According to the PEP 8 """ that ends a multiline docstring should be
# on a line by itself, unless it is a one-liner.
# 3) I removed a line between the while loop and the if-statement because it did not add any value and
# according to PEP 8 blank lines should be used sparingly within functions
# 4) I added some comments so that anyone could read and understand my code.
# However, all the code I wrote for these assignments is quite straight forwards, so I don't know
# how necessary these comments actually are.

# Q3.2P.7 ----------------------------------------------------------------------

# An example of a controversy about “the best way to do things” is how to handle packaging.
# While the Python Packaging Authority recommends pip, the tool for installing packages,
# it cannot handle non-Python dependencies.
# On the other hand, conda can manage both Python and non-Python dependencies.
# I personally think that using conda is more practicable, especially when working on projects that
# rely on other system-level libraries beyond Python.

# Q3.2P.8 ----------------------------------------------------------------------

# pip install pylint in concole

def diagnostic(insamples):
    """
    Returns two versions of Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples).
    Note that 'rhat' is more diagnostic than 'oldrhat' according to
    Gelman et al. (2014).

    Reference for preferred Rhat calculation (split chains) and number
        of effective sample calculation:
        Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2014).
        Bayesian data analysis (Third Edition). CRC Press:
        Boca Raton, FL

    Reference for original Rhat calculation:
        Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
        Bayesian Data Analysis (Second Edition). Chapman & Hall/CRC:
        Boca Raton, FL.


    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.

    Returns
    -------
    dict:
        rhat, oldrhat, neff, posterior mean, and posterior std for each variable.
        Prints maximum Rhat and minimum Neff across all variables
    """

    result = {}
    maxrhatsold = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float) * np.inf
    allkeys = {}
    keyindx = 0

    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}

            possamps = insamples[key]

            # Number of chains
            nchains = possamps.shape[-1]

            # Number of samples per chain
            nsamps = possamps.shape[-2]

            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])

            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps / 2), nchains * 2,))
            newc = 0
            for chain in range(nchains):
                possampsnew[..., newc] = np.take(np.take(possamps, np.arange(0, int(nsamps / 2)),
                                                         axis=-2), chain, axis=-1)
                possampsnew[..., newc + 1] = np.take(np.take(possamps, np.arange(int(nsamps / 2),
                                                            nsamps), axis=-2), chain, axis=-1)
                newc += 2

            # Index of variables
            varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

            # Reshape data
            alldata = np.reshape(possamps, (nvars, nsamps, nchains))

            # Mean of each chain for rhat
            chainmeans = np.mean(possamps, axis=-2)
            # Mean of each chain for rhatnew
            chainmeansnew = np.mean(possampsnew, axis=-2)
            # Global mean of each parameter for rhat
            globalmean = np.mean(chainmeans, axis=-1)
            globalmeannew = np.mean(chainmeansnew, axis=-1)
            result[key]['mean'] = globalmean
            result[key]['std'] = np.std(allsamps, axis=-1)
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            globalmeanextnew = np.expand_dims(
                globalmeannew, axis=-1)  # Expand the last dimension
            globalmeanextnew = np.repeat(
                globalmeanextnew, nchains * 2, axis=-1)  # For differencing
            # Between-chain variance for rhat
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain for rhat
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance for rhat
            totalestvar = (1. - (1. / nsamps)) * \
                          within + (1. / nsamps) * between
            # Rhat (original Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhatsold[keyindx] = np.nanmax(temprhat)  # Ignore NANs
            allkeys[keyindx] = key
            result[key]['oldrhat'] = temprhat
            # Between-chain variance for rhatnew
            betweennew = np.sum(np.square(chainmeansnew - globalmeanextnew),
                                axis=-1) * (nsamps / 2) / ((nchains * 2) - 1.)
            # Mean of the variances of each chain for rhatnew
            withinnew = np.mean(np.var(possampsnew, axis=-2), axis=-1)
            # Total estimated variance
            totalestvarnew = (1. - (1. / (nsamps / 2))) * \
                             withinnew + (1. / (nsamps / 2)) * betweennew
            # Rhatnew (Gelman-Rubin statistic from Gelman et al., 2013)
            temprhatnew = np.sqrt(totalestvarnew / withinnew)
            maxrhatsnew[keyindx] = np.nanmax(temprhatnew)  # Ignore NANs
            result[key]['rhat'] = temprhatnew
            # Number of effective samples from Gelman et al. (2013) 286-288
            neff = np.empty(possamps.shape[0:-2])
            for var in range(0, nvars):
                whereis = np.where(varindx == var)
                rho_hat = []
                rho_hat_even = 0
                rho_hat_odd = 0
                t = 2
                while ((t < nsamps - 2) & (float(rho_hat_even) + float(rho_hat_odd) >= 0)):
                    variogram_odd = np.mean(
                        np.mean(np.power(alldata[var, (t - 1):nsamps, :] - alldata[var,
                                0:(nsamps - t + 1), :], 2),
                                axis=0))  # above equation (11.7) in Gelman et al., 2013
                    rho_hat_odd = 1 - np.divide(variogram_odd,
                                2 * totalestvar[whereis])
                    # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_odd)
                    variogram_even = np.mean(
                        np.mean(np.power(alldata[var, t:nsamps, :] - alldata[var, 0:(nsamps - t), :], 2),
                                axis=0))  # above equation (11.7) in Gelman et al., 2013
                    rho_hat_even = 1 - np.divide(variogram_even,
                                 2 * totalestvar[whereis])
                    # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_even)
                    t += 2
                rho_hat = np.asarray(rho_hat)
                neff[whereis] = np.divide(nchains * nsamps,
                                          1 + 2 * np.sum(rho_hat))
                # Equation (11.8) in Gelman et al., 2013
            result[key]['neff'] = np.round(neff)
            minneff[keyindx] = np.nanmin(np.round(neff))
            keyindx += 1

    maxrhatkey = allkeys[np.argmax(maxrhatsnew)]
    maxrhatindx = np.unravel_index(np.argmax(result[maxrhatkey]['rhat']), result[maxrhatkey]['rhat'].shape)
    print("Maximum Rhat was %3.2f for variable %s at index %s" % (np.max(maxrhatsnew), maxrhatkey, maxrhatindx))
    minneffkey = allkeys[np.argmin(minneff)]
    minneffindx = np.unravel_index(np.argmin(result[minneffkey]['neff']), result[minneffkey]['neff'].shape)
    print("Minimum number of effective samples was %d for variable %s at index %s" % (
    np.min(minneff), minneffkey, minneffindx))
    return result

# Changes:
# 1) I removed comments like "# Initialize dictionary" as they are not very necessary.
# 2) I added a line before the first for loop, after the variables/lists/dicts were initialised.
# 3) I removed confusing comments like "# Geweke statistic?" as they do not increase the readability of the code.
# 4) I changed the variable names of variable "c" and "v" as they do not conform to snake_case naming style (invalid-name).
# 5) According to pylint, many lines were too long, which I shortend (for example line 14, 17, 39, 140, 154 of the function.)

# Q3.2P.9 ----------------------------------------------------------------------

def cheat(exercise_number):
    """ function returns the solution to Question 2 and 4 of Assignment 2. """

    if exercise_number == "Q2.2P.4" or exercise_number == "Assignment 2 Question 4" or exercise_number == "Question 4":

        # run function and output result
        from Assignment_2.Assignment_2_2 import unique_values
        vec = input("To compute the unique values, please enter your vector in the correct format here :")
        unique_values(vec)

        # print code of function
        import inspect
        print(inspect.getsource(unique_values))

    elif exercise_number == "Q2.2P.2" or exercise_number == "Assignment 2 Question 2" or exercise_number == "Question 2" :

        # run function and output result
        vec = input("To compute the weighted average, please enter your list here :")
        numeric_vec = np.array(vec)
        weighted_sum = 0

        for i in range(len(numeric_vec)):
            if i % 2 == 0:
                new_element = numeric_vec[i] * 2
            else:
                new_element = numeric_vec[i]
            weighted_sum += new_element
        weighted_average = weighted_sum / (np.size(numeric_vec) * 1.5)

        print(" The weighted average is", weighted_average)

        # print code of function
        print(
            """vec = input("To compute the weighted average, please enter your list here :")
        numeric_vec = np.array(vec)
        weighted_sum = 0

        for i in range(len(numeric_vec)):
            if i % 2 == 0:
                new_element = numeric_vec[i] * 2
            else:
                new_element = numeric_vec[i]
            weighted_sum += new_element
        weighted_average = weighted_sum / (np.size(numeric_vec) * 1.5)

        print(" The weighted average is", weighted_average)
        """
        )

    else:
        print("Unfortunatly, this function is unable to answer your selected question.",
              "\n", "Please try a different question!")

# link to githup
#