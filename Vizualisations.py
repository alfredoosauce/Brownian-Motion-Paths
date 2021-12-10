import numpy as np
import matplotlib.pyplot as plt


def mainCalculation():
    noPaths = 25
    noSteps = 500
    T = 1
    sigma = 0.4
    mu = 0.05
    S_0 = 150

    Paths = pathGeneratorGBMABM(noPaths, noSteps, T, sigma, mu, S_0)
    timeGrid = Paths['time']
    X = Paths['X']
    S = Paths['S']

    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('X(t)')

    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('S(t)')
    plt.show()

def pathGeneratorGBMABM(noPaths, noSteps, T, sigma, mu, S_0):
    np.random.seed(1)  # For stability to run tests

    Z = np.random.normal(0.0, 1.0, [noPaths, noSteps])  # 2D array with 25*500 random N(0,1) variables. Size path*steps
    X = np.zeros([noPaths, noSteps+1])  # Array of size paths*steps filled w/ zeros.
    S = np.zeros([noPaths, noSteps+1])  # Same ^. Also steps+1 needed to have the x_0 place
    time = np.zeros([noSteps + 1])
    X[:, 0] = np.log(S_0)  # Set log(S_0) for the 0th Column of all rows.
    dt = T / float(noSteps)  # Calculate step size

    for i in range(0, noSteps):
        if noPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # Normalize all values of ith Column of RV's

        #  Calculate the X(t + dt) =
        X[:, i + 1] = X[:, i] + (mu - 0.5 * sigma ** 2) * dt + (sigma * np.power(dt, 0.5) * Z[:, i])
        time[i + 1] = time[i] + dt  # Here we are recording the time change from i=0 --> i=dt

    # Exponent of ABM
    S = np.exp(X)
    # Return dictionary with time, X, S computed values
    paths = {"time": time, "X": X, "S": S}  # Maps key to definition: e.g., print(paths['S']) out S
    return paths

mainCalculation()
