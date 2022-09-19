from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/1155539/how-do-i-generate-a-poisson-process
def homo_poisson_process(lamb: float, T: float, srate: float) -> NDArray[np.float64]:    
    """generate homogeneous poisson process

    Args:
        lamb (float): lambda, sec^-1
        T (float): total sampling time, sec
        srate (float): sampling rate, Hz

    Returns:
        NDArray[np.float64]: trajectory with floor(T*srate) length
    """

    train = np.zeros(np.floor(T*srate).astype(int))
    # check if lamb>0
    if lamb <= 0: 
        return train
    
    # generate intervals, 1/beta == lamb/srate 
    intervals = []
    while np.sum(intervals) <= len(train):
        intervals.append(np.random.exponential(1.0*srate/lamb))

    # create fire train
    current = 0
    while np.ceil(current+intervals[0]) < len(train):
        current += intervals[0]
        train[np.ceil(current).astype(int)] = 1.0
        intervals.pop(0) 

    return train

# https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasB.pdf, Algorothm 7
def nonhomo_poisson_process(Lamb: NDArray[np.float64], srate: float) -> NDArray[np.float64]:
    """generate nonhomogeneous poisson process

    Args:
        Lamb (NDArray[np.float64]): Lambda, Dimensionless
        srate (float): sampling rate, Hz

    Returns:
        NDArray[np.float64]: trajectory with len(Lamb) length
    """

    train = np.zeros_like(Lamb)
    # check if lamb>0
    if np.any(np.array(Lamb) <= 0): 
        return train
    
    # generate homo poission process lamb_u = max(Lamb)
    lamb_u = np.max(np.array(Lamb))

    # generate intervals, 1/beta == lamb_u/srate 
    intervals = []
    while np.sum(intervals) <= len(train):
        intervals.append(np.random.exponential(1.0*srate/lamb_u))

    # create fire train, accept with possibility Lamb(t)/lamb_u
    current = 0
    while np.ceil(current+intervals[0]) < len(train):
        current += intervals[0]
        # simply linear interpolation
        if (Lamb[np.ceil(current).astype(int)]+(current-np.ceil(current))*(Lamb[np.ceil(current).astype(int)+1]-Lamb[np.ceil(current).astype(int)]))/lamb_u >= np.random.uniform(0,1):
            train[np.ceil(current).astype(int)] = 1.0
        intervals.pop(0) 

    return train

if __name__ == "__main__":
    Lamb = np.ones(10*10000)*10
    # Lamb[:5*10000] += 17
    x = nonhomo_poisson_process(Lamb,10000)

    plt.plot(x)
    plt.show()

    print(np.sum(np.array(x)==1))
