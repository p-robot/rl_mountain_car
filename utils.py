# utils.py
#!/usr/bin/python
"""
Module of utilities for project applying adaptive management to foot-and-mouth disease control.  
"""

import copy, time 
import numpy as np, pandas as pd
import scipy.spatial as sp
import pickle

def getpickle(filename):
    """
    Given a path to a picked item, unpickle the item, saves it to a dict
    """
    try:
        f = open(filename, "r")
    except IOError:
        print('cannot open', filename)
    else:
        p = cPickle.Unpickler(f)
        dictionary = p.load()
        f.close()
        return(dictionary)

def setpickle(filename, names, scope):
    """
    Save the local variables in the list 'names' to the pickled file named 'filename'.  
    set 'scope' to locals()
    """
    print('circ4ksarsa' in scope)
    try:
        f = open(filename, "w")
    except IOError:
        print('cannot open', filename)
    else:
        p = cPickle.Pickler(f)
        dictionary = {}
        for n in names:
            dictionary[n] = scope[n]
        p.dump(dictionary)
        f.close()

def getparameters(file_scenario, file_control_choices, \
    file_vaccination_schedule, file_system, file_infection):
    
    """
    Import parameters to a dictionary.  
    
    Note: pandas package is not used so that it can be used on the HPCU at PSU.  
    """
    
    # Import status and landscape parameters 
    # make dtype None so that the function will guess the type of the object
    
    scenario = np.genfromtxt(file_scenario, 
        delimiter = ",", names = True, dtype = None) 
    
    dict_landscape = dict()
    for n in scenario.dtype.names:
        dict_landscape[n] = np.asarray(scenario[n])
        dict_landscape[n].shape = (1,)*(1 - dict_landscape[n].ndim) + \
            dict_landscape[n].shape
    
    x = dict_landscape['x']; y = dict_landscape['y']
    dict_landscape['coords'] = zip(x, y)
    
    # Import infection parameters
    infection = np.genfromtxt(file_infection, \
        delimiter = ",", names = True, dtype=None)
    
    for n in infection.dtype.names:
        dict_landscape[n] = infection[n]
    
    # Calculate the susceptibility and transmissibility of each farm
    suscept_sheep = dict_landscape['suscept_sheep']
    suscept_cows = dict_landscape['suscept_cows']
    transmiss_sheep = dict_landscape['transmiss_sheep']
    transmiss_cows = dict_landscape['transmiss_cows']
    n_cows = dict_landscape['n_cows']
    n_sheep = dict_landscape['n_sheep']
    
    # Import system parameters
    system = np.genfromtxt(file_system, \
        delimiter = ",", names = True, dtype=None)
    
    dict_system = dict()
    for n in system.dtype.names:
        dict_system[n] = system[n].flatten()[0]
    
    # Import control parameters
    if isinstance(file_control_choices, str):
        control = dict()
        control_choices = np.genfromtxt(file_control_choices, \
            delimiter = ",", names = True, dtype=None)
        
        for i, n in enumerate(control_choices):
            control[i+1] = dict()
            control[i+1]['method'] = n[0]
            control[i+1]['inner_radius'] = n[1]
            control[i+1]['outer_radius'] = n[2]
        
        vaccination_schedule = np.genfromtxt(file_vaccination_schedule,
            delimiter = ",", names = True, dtype=None)
        
        vacc_sch = dict()
        for n in vaccination_schedule:
            vacc_sch[n[0]] = n[1]
    
        dict_control = dict()
        dict_control['control_choices'] = control
        dict_control['control_switch_times'] = [7] + \
            3*np.arange(dict_system['max_time']/3)
        dict_control['vaccine_horizons'] = vacc_sch
    else:
        dict_control = dict()
    
    return(dict_landscape, dict_control, dict_system)


def Neighbourhood(D, outer, inner = 0):
    """
    Return boolean array of whether distances (in D) are within a particular 
    combination of radii (inner, outer) and no distance is less than 'inner'
    (this removes the problem of when a susceptible feedlot may be in the
    vaccination ring of one IP but inside the inner radius of another IP -
    vaccination candidates should only be those inside the 'combined cloud'
    created by combining the annulii around all IPs).  
    
    INPUT
    
    D : distance matrix of dimension 
        (sum(detected_mask) x sum(not_detected_mask))
    
    outer: 
        outer radius of vaccination ring
    
    inner: 
        inner radius of vaccination ring
    
    
    OUTPUT
    
    candate: boolean np.array of length sum(not_detected_mask) having True
         values for which feedlots are candidates for vaccination.  More
         specifically, True for which feedlots 1) don't have fmd infected
         status, 2) are within the vaccination ring for SOME infected premises,
         and 3) are within the inner vaccination ring for NONE of the infected
         premises.  
    
    Radii cut-off are inclusive - on the border means the feedlot is included
    in the ring.  
    
    Note: 
    The notation used in np.empty_like is new in Numpy version 1.6.0 so 
    therefore any computer or HPCU must set python to 2.7.3 or later.
    
    EXAMPLE
    
    import numpy as np
    A = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]])
    Neighbourhood(A, outer = 2.1, inner = 0)
    
    """
    
    if outer < inner:
        raise ValueError("Outer radius must be larger than inner radius")
    
    # Generate an empty array the same size as D
    nbhd = np.empty_like(D)#, dtype = bool) 
    
    # Find all distances within the radii given
    for i, d in enumerate(D):
        nbhd[i] = (d <= outer) & (d >= inner)
    # Is there a more efficient ways to do this? map() or a list comp? 
    
    # Make sure no feedlot is within 'inner' radius to any infected premises
    too_close = ((D < inner) & (D != 0)).any(axis = 0)
    too_close = np.array(too_close).flatten()
    
    # Change type to np.array instead of matrix 
    # (inherits type from D, a matrix).  
    nbhd = np.array(nbhd, dtype = bool)
    
    # Combine these two conditions
    candidates = nbhd.any(axis = 0)
    candidates[too_close] = False
    
    return(candidates)


def NeighbourhoodDetail(D, outer, inner = 0):
    """
    Extension of the 'Neighbourhood' function above but giving the ID of the 
    IP farm that's closest to each suscept farm that's earmarked for control.  
    
    Return boolean array of whether distances (in D) are within a particular 
    combination of radii (inner, outer), and also the row number where such a 
    condition is satisfied.  
    
    {True : (D_{i,j} > inner) & (D_{i,j} < outer) }
    {i : (D_{i,j} > inner) & (D_{i,j} < outer) }
    
    INPUT
    
    D : distance matrix of dimension 
        (sum(detected_mask) x sum(not_detected_mask))
    
    outer: 
        outer radius of vaccination ring
    
    inner: 
        inner radius of vaccination ring
    
    OUTPUT
    
    candidates: boolean np.array of length sum(not_detected_mask) having True
         values for which feedlots are candidates for vaccination.  More
         specifically, True for which feedlots 1) don't have fmd infected
         status, 2) are within the vaccination ring for SOME infected premises,
         and 3) are within the inner vaccination ring for NONE of the infected
         premises.  
    
    mindist2ip
    which_ip_min (how are ties dealt with in this case?  )
    
    
    Radii cut-off are inclusive - on the border means the feedlot is included
    in the ring.  
    
    Note: 
    The notation used in np.empty_like is new in Numpy version 1.6.0 so 
    therefore any computer or HPCU must set python to 2.7.3 or later.
    
    EXAMPLE
    
    import scipy.spatial as sp
    import numpy as np
    from am4fmd import utils
    
    # Generate points on a grid
    x = np.linspace(-2, 2, 11)
    XX, YY = np.meshgrid(x, x)
    XX = XX.ravel(order = 'C'); YY = YY.ravel(order = 'C')
    coords = zip(XX, YY)
    
    # Generate a vector of the infection status of farms (anything > 0 is inf)
    status = np.zeros(len(coords))
    cntr = [60, 64]
    status[cntr] = 1
    
    # Calculate a distance matrix between points
    D = sp.distance.cdist(coords, coords)
    
    # Subset the distance matrix to be infected (rows) - susceptible (cols)
    D_inf_to_sus = D[cntr,:]
    D_inf_to_sus = np.delete(D_inf_to_sus, cntr, 1)
    
    # If using iPython, time the implementation
    %timeit utils.NeighbourhoodDetail(D_inf_to_sus,outer=1.9,inner=1.1)
    
    # Calculate the 'neighbours' within each infected premises
    ca, mi, wh = utils.NeighbourhoodDetail(D_inf_to_sus,outer=1.9, inner=1.1)
    
    # The output is in terms of susceptible/infected farms, so need these:
    sus = np.where(status == 0)[0]
    inf = np.where(status > 0)[0]
    
    # Plot farms, those in the 'neighbourhood' are in orange, the numbers show
    # index of the which of the infected farm that designated them as a cull
    # candidate, numbers in blue show the minimum distance to any IP
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    
    # Green for each farm
    ax.scatter(XX, YY, c = 'green')
    
    # Red for infected farms
    ax.scatter(XX[cntr], YY[cntr], c = 'red')
    
    # Orange for those in the 'neighbourhood'
    ax.scatter(XX[sus[ca]], YY[sus[ca]], c = 'orange', \
        s = 200, lw = 0, marker = 'o')
    
    for a, b, c in zip(XX[sus[ca]], YY[sus[ca]], [str(x) for x in wh]):
        ax.text(a, b, c, va = 'center', ha = 'center')
    
    strings = ['\n\n'+str(np.round(x,2)) for x in mi]
    for a, b, c in zip(XX[sus[ca]], YY[sus[ca]], strings):
        ax.text(a, b, c, va = 'center', ha = 'center', \
        color = 'blue', fontsize = 8)
    
    plt.show()
    
    """
    
    nbhd = (D <= outer)&(D >= inner)
    candidates = nbhd.any(axis = 0)
    
    # Make sure no feedlot is within 'inner' radius to any infected premises
    too_close = ((D < inner) & (D != 0)).any(axis = 0)
    
    # Combine these two conditions
    candidates[too_close] = False
    
    which_ip_min = D[:, candidates].argmin(axis = 0)
    mindist2ip = D[:, candidates].min(axis = 0)
    #maxdist2ip = D[:, candidates].max(axis = 0)
    
    return(candidates, mindist2ip, which_ip_min)

def calc_grid_probs(grid, grid_n_x, grid_n_y, Suscept, kernel_function, \
    d_perimeter_x, d_perimeter_y):
    """
    Calculate the probability of infection from one grid square to the next,
    optimized for Python.  
    
    Notes
    -----
    
    Could also use pairwise chebyshev distances... instead of
    np.subtract.outer()
    
    
    Parameters
    ----------
        grid: np.array
    
        grid_n_x: np.array
    
        grid_n_y: np.array
    
        Suscept: np.array
    
        kernel_function: function
    
        d_perimeter_x : float
        
        d_perimeter_y : float
        
        
        
    Returns
    -------
        
        MaxRate
        
        Num
        
        first_
        
        last_
        
        max_sus
        
        Dist2all
        
        KDist2all
    
    
    Updates
    -------
    
    # 6th Oct 2014 Changed use of grid.max()+1 to grid_n_x*grid_n_y, WP
    """
    
    # Find the farm indices per grid square
    farms_per_grid = [np.where(grid == i)[0] for i in range(grid_n_x*grid_n_y)]
    
    # Maximum susceptibility of a feedlot per grid square
    max_sus = np.asarray([np.max(np.append(Suscept[farms_per_grid[index]], 0)) for index in range(grid_n_x*grid_n_y)])
    
    # Find the number of farms per grid square
    Num = np.array([len(i) for i in farms_per_grid], dtype = int)
    
    # First farm in grid, -1 if the list is empty
    first_ = [np.min(i) if len(i) > 0 else -1 for i in farms_per_grid]
    first_ = np.asarray(first_).astype(int)
    
    # Convert np.ana to -1; convert whole array to int types
    #first_[np.isnan(first_)] = -1; first_ = first_.astype(int)
    
    # Last farm in grid, -2 if the list is empty
    last_ = [np.max(np.append(index, -2)) for index in farms_per_grid]
    last_ = np.asarray(last_).astype(int)
    
    # X and Y position of each grid square
    gridx = np.repeat(np.arange(grid_n_x), grid_n_y) # a flattening of meshgrid
    gridy = np.tile(np.arange(grid_n_y), grid_n_x)
    
    qq = np.abs(np.subtract.outer(gridx, gridx)) - 1
    qq[qq < 0] = 0
    
    pp = np.abs(np.subtract.outer(gridy, gridy)) - 1
    pp[pp < 0] = 0
    
    ppsq = (d_perimeter_x*pp/float(grid_n_x))**2
    qqsq = (d_perimeter_y*qq/float(grid_n_y))**2
    
    # multiply by the Max susceptibility of the target grid square.
    Dist2all = np.add(ppsq, qqsq)
    KDist2all = kernel_function(Dist2all)
    MaxRate = np.multiply(KDist2all, np.tile(max_sus, (grid_n_x*grid_n_y, 1)))
    
    # All grids with no farms should have infinite rate of input infection
    # All grids with no farms should have infinite rate of susceptibility
    MaxRate[Num == 0] = np.inf
    MaxRate[:, Num == 0] = np.inf
    
    # Set rate from one grid to itself to infinity.  
    np.fill_diagonal(MaxRate, np.inf)
    
    return(MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all)


def calc_grid_probs2(grid, grid_n_x, grid_n_y, Suscept, kernel_function,\
    d_perimeter_x, d_perimeter_y):
    """
    Calculate the probability of infection from one grid square to the next.
    (quicker than version above)
    
    Notes
    -----
    Grid and Suscept MUST be ordered according to grid number!  
    
    Could also use pairwise chebyshev distances (scipy.spatial.distance.cdist)
    instead of using ogrid etc (below) if the x/y lengths of the grid cells
    are the same.  
    
    
    Parameters
    ----------
        grid: np.array
    
        grid_n_x: np.array
    
        grid_n_y: np.array
    
        Suscept: np.array
    
        kernel_function: function
    
        d_perimeter_x : float
        
        d_perimeter_y : float
        
        
    Returns
    -------
        
        MaxRate
        
        Num
        
        first_
        
        last_
        
        max_sus
        
        Dist2all
        
        KDist2all
    
    
    Example
    -------
    # Calculate grid-level probabilities of infection for the UK data
    
    import epi_speed_test.py.fmd as fmd
    kernel = fmd.UKKernel
    out = calc_grid_probs(grid, 100, 100, np.random.rand(len(grid)), kernel, \
        np.ptp(demo.easting.values), np.ptp(demo.northing.values))
    
    
    Updates
    -------
    
    # 6th Oct 2014 Changed use of grid.max()+1 to grid_n_x*grid_n_y, WP
    # Feb 2017 Changed to use np.ogrid etc, WP
    """
    
    NG = grid_n_x*grid_n_y
    
    # Find the farm indices per grid square
    fpg = [np.where(grid == i)[0] for i in range(NG)]
    # Perhaps be made quicker by only looking at the grids with farms in them.
    
    # Maximum susceptibility of a feedlot per grid square (0 for empty grid)
    max_sus = np.asarray([np.max(Suscept[x]) if len(x) > 0 else 0 for x in fpg])
    
    # Find the number of farms per grid square
    Num = np.array([len(i) for i in fpg])
    
    # First farm in grid, -1 if the list is empty
    first_ = np.asarray([x[0] if len(x) > 0 else -1 for x in fpg])
    
    # Last farm in grid, -2 if the list is empty
    last_ = np.asarray([x[-1] if len(x) > 0 else -2 for x in fpg])
    
    NCOL = float(grid_n_x); NROW = float(grid_n_y)
    HEIGHT = d_perimeter_y/NROW; WIDTH = d_perimeter_x/NCOL
    
    # Calculate the minimum sq-distance between each grid square
    # X and Y position of each grid square
    Y, X = np.ogrid[0:NROW*NCOL, 0:NROW*NCOL]
    HDIST = np.abs(Y%NCOL - X%NCOL) - 1
    VDIST = np.abs(Y/NCOL - X/NCOL) - 1
    HDIST[HDIST < 0] = 0; VDIST[VDIST < 0] = 0
    Dist2all = np.add((HDIST*WIDTH)**2, (VDIST*HEIGHT)**2)
    
    KDist2all = kernel_function(Dist2all) # very slow
    
    # Could perhaps use Numpy broadcasting for this ... 
    MaxRate = np.multiply(KDist2all, np.tile(max_sus, (grid_n_x*grid_n_y, 1)))
    
    # All grids with no farms should have infinite rate of input infection
    # All grids with no farms should have infinite rate of susceptibility
    MaxRate[Num == 0] = np.inf
    MaxRate[:, Num == 0] = np.inf
    
    # Set rate from one grid to itself to infinity.  
    np.fill_diagonal(MaxRate, np.inf)
    
    return(MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all)


def dict2df_2d(Q, trace, statenames):
    """
    Transform a dictionary with keys as a tuple of two-state variables into a pandas dataframe
    """
    
    n_acts = len(list(Q.values())[0])
    act_names = ['a'+str(x) for x in range(n_acts)]
    statenames.extend(act_names)
    
    # Output the value and visitation matrices to csv files.  
    states = pd.DataFrame(list(Q.keys()))
    values = pd.DataFrame(list(Q.values()))
    q = pd.concat((states, values), axis = 1)
    
    ind = q.iloc[:,2:(2+n_acts)].idxmax(axis = 1)
    q.columns = statenames
    
    t = np.array(list(trace.keys()))
    n = np.array(list(trace.values()))
    
    # Combine them... BE CAREFUL WITH THIS AS IT'S NOT USING AN INDEX
    q['visits'] = n
    q['qmax'] = np.max(q.iloc[:,2:(2+n_acts)], axis = 1)
    q['astar'] = ind
    q['astar_name'] = q.iloc[:,2:(2+n_acts)].idxmax(axis = 1)
    
    return q


def dict2df_2d_notrace(Q, visits, statenames = ['ninf', 'area']):
    """
    Transform a dictionary with keys as a tuple of 
    two-state variables into a pandas dataframe
    
    No trace is passed.  
    
    visits : dict or int
    """
    
    if isinstance(visits, dict):
        visit_states = pd.DataFrame(list(visits.keys()))
        visit_states.columns = statenames
        visit_states['visits'] = list(visits.values())
    
    n_acts = len(list(Q.values())[0])
    act_names = ['a'+str(x) for x in range(n_acts)]
    statenames.extend(act_names)
    
    # Output the value and visitation matrices to csv files.  
    states = pd.DataFrame(list(Q.keys()))
    
    if isinstance(visits, dict):
        values = pd.DataFrame(list(Q.values()))
    else:
        values = pd.DataFrame([[np.mean(r) for r in v] for v in list(Q.values())])
    
    q = pd.concat((states, values), axis = 1)
    
    ind = q.iloc[:,2:(2+n_acts)].idxmin(axis = 1)
    q.columns = statenames
    
    # Combine them... BE CAREFUL WITH THIS AS IT'S NOT USING AN INDEX
    q['qmin'] = np.min(q.iloc[:,2:(2+n_acts)], axis = 1)
    q['astar'] = ind
    q['astar_name'] = q.iloc[:,2:(2+n_acts)].idxmin(axis = 1)
    
    if isinstance(visits, dict):
        q = pd.merge(q, visit_states)
    else: 
        q['visits'] = visits
    
    return q


def egreedy(Qs, actions, epsilon):
    """
    Epsilon greedy policy.  
    
    This function represents an epsilon greedy policy used to choose actions
    given a state.  The action with the maximum value in the value function
    is chosen, but a random action is chosen with probability epsilon (an 
    attribute of the agent).  Ties in maximum value function between 
    different actions are resolved randomly.  
    
    Arguments
    ---------
    Qs: np.array
        Current value function
    
    actions : list
        List of actions
    
    epsilon : double
        Learning parameter
    
    Returns
    -------
    The action to take given the current state according to an egreedy algorithm.  
    """
    
    # Find the action with the maximum value
    action = np.where(Qs == np.max(Qs))[0]
    
    # Deal with ties 
    if len(action) > 1:
        np.random.shuffle(action)
    
    if len(action) == 0:
        action = [0]
    
    action = action[0]
    
    # Choose a random action with probability epsilon
    if np.random.rand() < epsilon:
        action = np.random.randint(len(actions))
    
    return actions[action]


def rmse(a, b):
    """
    Calculate the root mean square error between two arrays, a and b
    
    Args:
        a, b: np.array
        Arrays of the same length, the root mean square error between these is 
        desired
    
    Returns:
        Root mean squared error between the two arrays
    """
    
    error_sq = (a - b)**2
    return np.sqrt(np.mean(error_sq))


def Coords2Grid(x, y, perim_x, perim_y, grid_n_x, grid_n_y):
    """
    Determine grid locations of farms from x, y coordinates
    Returns the grid of farms in row-major order.  
    
    Parameters
    ----------
        x : numpy.array; float
            x coordinates of farms
        
        y : numpy.array; float
            y coordinates of farms
        
        perim_x: float
            width of landscape in x-dimension
        
        perim_y: 
            width of landscape in y-dimension
        
        grid_n_x: 
            number of desired grid squares in the horizontal direction
            
        grid_n_y: 
            number of desired grid squares in the vertical direction
        
    Returns
    -------
        grid_x : numpy.array
            
            
        grid_y : numpy.array
            
            
        grid : numpy.array
            
    """
    # Create boundaries for the grid squares
    xbins = np.linspace(start = np.min(x), 
        stop = np.min(x) + perim_x, 
        num = grid_n_x+1)
    
    ybins = np.linspace(start = np.min(y), 
        stop = np.min(y) + perim_y, 
        num = grid_n_y+1)
    
    # Find the x- and y-specific grid squares
    grid_x = np.digitize(x, xbins, right = True)
    grid_x[grid_x == 0] = 1
    
    grid_y = np.digitize(y, ybins, right = True)
    grid_y[grid_y == 0] = 1
    
    # Adjust for python index numbering
    grid_x = grid_x - 1
    grid_y = grid_y - 1
    
    # Convert from 2D coords to a 1D index
    grid = np.ravel_multi_index((grid_x, grid_y), 
        dims = (grid_n_x, grid_n_y), 
        order = 'C')
    
    return(grid_x, grid_y, grid)


def Coords2Grid2(x, y, grid_n_x, grid_n_y):
    """
    Simpler [updated] version of Coords2Grid
    
    Returns the grid of farms in row-major order.  
    
    Parameters
    ----------
        x : np.array; float
            x coordinates of farms
        
        y : np.array; float
            y coordinates of farms
        
        grid_n_x: int
            number of desired grid squares in the horizontal direction
            
        grid_n_y: int
            number of desired grid squares in the vertical direction
        
    Returns
    -------
        grid : np.array
            row-major grid number of farm
    
    Example
    -------
    from matplotlib import pyplot as plt
    import numpy as np
    
    # Generate a 4 x 4 grid over a landscape of 10 farms
    N = 100; dim = 20.
    xx = np.random.rand(N)*dim; yy = np.random.rand(N)*dim
    grid = Coords2Grid(xx, yy, 4, 4)
    
    plt.scatter(xx, yy, color = 'grey')
    for x, y, g in zip(xx, yy, grid):
        plt.text(x, y, g)
    
    for x, y, g in zip(xx, yy, gx):
        plt.text(x - 0.5, y, g, color = 'blue')
    
    for x, y, g in zip(xx, yy, gy):
        plt.text(x + 0.5, y, g, color = 'red')
    
    plt.show()
    
    """
    
    # Create boundaries for the grid squares
    xbins = np.linspace(
        start = np.min(x), 
        stop = x.max(), 
        num = grid_n_x + 1)
    
    ybins = np.linspace(
        start = np.max(y), 
        stop = np.min(y), 
        num = grid_n_y + 1)
    
    # Find the x- and y-specific grid squares
    grid_x = np.digitize(x, xbins, right = True)
    
    grid_y = np.digitize(y, ybins, right = False)
    
    # Adjust top and bottom x/y grid values due to numerical rounding
    grid_x[grid_x == 0] = 1
    grid_y[grid_y == 0] = 1
    
    # Adjust for python index numbering
    grid_x = grid_x - 1
    grid_y = grid_y - 1
    
    # Convert from 2D coords to a 1D index (row-major order)
    grid = np.ravel_multi_index((grid_x, grid_y), \
        dims = (grid_n_x, grid_n_y), order = 'F')
    
    return grid


def per_farm_output(sarsa_list):
    """
    This takes the full output from an outbreak [(S,A,R,S) tuples] as a list
    and creates a data frame of per-time-step events.  
    
    # Per time step:
    #   Stack the per-farm arrays into a long array.  
    #   Stack the scalars into an array.  
    #   Concatenate into a vertical dataframe
    # Create dataframe
    # Create titles
    """
    
    T = len(sarsa_list)
    for t, sarsa in enumerate(sarsa_list):
        #print("Time step: ", t)
        # (s, a, r, s, a)
        s = sarsa[0]
        a = sarsa[1]
        r = sarsa[2]
        next_s = sarsa[3]
        
        # Extract the state of each farm # s.n_cattle_original
        row = np.hstack((s.status, s.n_cattle, s.n_carcass, s.n_disposed, \
            s.n_vacc.flatten(), s.n_i, s.n_i_pseudo, \
            s.being_culled, s.being_vacc, s.being_disposed))
        
        # Scalar values... 
        row_scalars = np.hstack((t, s.n_inf, s.area, r, a, a.ind, \
            s.vacc_used, s.ci)) #, s.n_notified))
        
        full_row = np.hstack((row_scalars, row))
        
        if (t == 0):
            full = full_row
        else:
            full = np.vstack((full, full_row))
    
    # Create pandas dataframe
    full = pd.DataFrame(full)
    
    N = s.nfarms
    # Add titles to the dataframe
    states = ['status'+str(i) for i in range(N)]
    n_cattle = ['n_cattle'+str(i) for i in range(N)]
    n_carcass = ['n_car'+str(i) for i in range(N)]
    n_dispose = ['n_dis'+str(i) for i in range(N)]
    #n_cattle_o = ['n_cattle_o'+str(i) for i in range(s.n_feedlots)]
    n_vacc = ['vday'+str(i)+"-"+str(j) for i in range(s.ci) \
        for j in range(N)]
    n_i = ['n_i'+str(i) for i in range(N)]
    n_i_p = ['n_i_ps'+str(i) for i in range(N)]
    bg_cull = ['bg_cull'+str(i) for i in range(N)]
    bg_vacc = ['bg_vacc'+str(i) for i in range(N)]
    bg_dis = ['bg_dis'+str(i) for i in range(N)]
    
    scalars = ['t', 'n_inf', 'area', 'r', 'a', 'ai', 'vacc_used', \
        'ci']#, 'n_not']
    
    full.columns = np.hstack((scalars, states, n_cattle, n_carcass, n_dispose, \
        n_vacc, n_i, n_i_p, bg_cull, bg_vacc, bg_dis))
    
    return full

def save_state(s, location):
    """
    Save a State object to file as a Python pickle object (or json?)
    
    # Scalars
    self.vacc_used
    self.ci
    self.hull
    self.area
    _n_notified
    terminal
    
    # Numpy 2D arrays
    self.n_vacc = np.array
    self.n_i = np.array
    self.n_i_pseudo = np.array
    
    # Calculated at instantiation (so no need to save these)
    # self.n_inf
    # self.n_farms
    
    Notes
    -----
    
    Note 1
    
    Tried using json.dumps but so many of the attributes of the State are numpy arrays of np.int64
    integers and json doesn't like this.  Tried the following but needed a quick solution so moved
    to pickle.  
    
    import json
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError
    j = json.dumps(dictionary, default = default)
    
    
    Note 2
    Can open the file using the following: 
    
    import pickle
    with open(inputfilename, 'rb') as handle:
        b = pickle.load(handle)
    
    W. Probert, 2018
    """
    
    dictionary = {}
    
    # Save scalars into a dictionary
    scalars = ['vacc_used', 'ci', 'hull', 'area', '_n_notified', 'terminal']
    for scalar in scalars:
        dictionary[scalar] = getattr(s, scalar)
    
    
    # Save 1-D arrays into a dictionary
    arrays1d = ['status', 'n_cattle', 'n_cattle_original', 'n_sheep', 'n_carcass', 'n_disposed', \
        'being_culled', 'being_vacc', 'being_disposed', 'when_reported', 'once_an_ip', \
        'dist2IP', 'whichIP', 'notified']
    
    for array1d in arrays1d:
        dictionary[array1d] = getattr(s, array1d)
    
    # Numpy 2D arrays
    arrays2d = ['n_vacc', 'n_i', 'n_i_pseudo']
    
    for array2d in arrays2d:
        dictionary[array2d] = getattr(s, array2d)
    
    with open(location, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    return 1


def dist_sqeuclidean1(x1, x2):
    """
    Calculate the squared Euclidean distance between two points
    """
    
    return (x1[1] - x2[1])*(x1[1] - x2[1]) + (x1[0] - x2[0])*(x1[0] - x2[0])


def dist_sqeuclidean(x1, x2):
    """
    Calculate the squared Euclidean distance between two points
    (Quicker than not using powers, see above)
    (Quicker again if they're just integers)
    """
    
    return (x1[1] - x2[1])**2 + (x1[0] - x2[0])**2



def dict2array(x, y, Q, p_res, v_res):
    """
    Convert a dictionary to an array
    for use in functions such as matplotlib.pyplot.imshow
    
    
    INPUT
        x   : np.array
            x axis of the array
        
        y   : np.array
            y axis of the array
        
        Q   : dict
            dictionary with entries as tuples of (x_i, y_j)
    
    OUTPUT
        out : np.array
            an array of dimension [len(x) x len(y)]
    
    """
    
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[xx.ravel(), yy.ravel()]
    
    result = []
    for xi in x:
        for yi in y:
            if( (xi,yi) in Q ):
                result.append(-np.max(Q[(xi,yi)]))
            else:
                result.append(0)

    out = np.array(result).reshape(p_res, v_res, order = 'F')
    out[np.isnan(out)] = 0
    return(xx, yy, out)
