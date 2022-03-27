
import numpy as np
import rli, utils

class CarState(rli.State):
    pass


class MountainCarSim(rli.Simulation):
    """
    Simulation object for the mountain car problem.  
    
    See the Simulation object in the rli module for a description of the 
    methods.  The only custom method in this subclass is collect_data.  
    
    Parameters
    ----------
        Agent : 
            Agent to be used in the simulation
        Environment : 
            Environment to be used in the simulation
        max_time : 
            Maximum wall time (in hours) for any simulation
        
        
        outlist: a list that saves all states, rewards, and actions in the 
            simulation so far
        durations: a list of durations of each simulation
        current_s: the current state
        current_a: the current action
        terminal_state: the placeholder for the terminal state
    
    Methods
    -------
        collect_data: save data throughout the simulation
        
        Further methods defined in the rli.Simulation parent class.  
    """
    
    def __init__(self, Agent, Environment, max_time, full = False, **kwargs):
        self.p_res = 20
        self.v_res = 20
        
        # Discretize the different state variables
        self.pbins = np.linspace(-1.2, 0.6, num = self.p_res)
        self.vbins = np.linspace(-0.07, 0.07, num = self.v_res)
        
        self.outlist = []
        self.full = full
        
        super(self.__class__, self).__init__(Agent, Environment, max_time, \
            **kwargs)
        
    def full2obs(self, current_s):
        """Translate the full state of the environment to the observed state.
        
        In several reinforcement learning tasks the agent will not see the 
        full state of the environment but a reduced form of the state, ie an 
        observed state.  This function translates the full state of the 
        environment (that is output by the environment step function) to the 
        state that is observed by the agent (and input to the agent step 
        function).  By default, the full state and observed state are the same
        so the user need not change this function.  
        
        Args:
            s: current full state of the environment
            
        Returns
            The state observed by the agent (a function of the full state of 
            the environment).  
        """
        
        if (current_s == self.terminal_state):
            s = (-5555, -5555)
        else:
            indp = np.digitize([current_s[0]], self.pbins, \
                right = True)[0]
            
            indv = np.digitize([current_s[1]], self.vbins, \
                right = True)[0]
            
            s = (self.pbins[indp], self.vbins[indv])
            
        return s
        
    def collect_data(self, current_s, current_a, reward, next_s):
        """
        Save data at each step of the simulation.  
        
        This method saves the current state, action, reward and next state at
        each time step of the simulation.  The list 'current_trial' keeps the 
        progress of the current trial, and the list 'outlist' keeps the history
        of all completed trials.  Once a terminal state is reached, the 
        attribute current_trial is reset to an empty list and the duration of 
        the completed trial is saved to the 'durations' attribute.  
        Note that if one stops a simulation without getting to the terminal 
        state then the history of all previously completed simulations will be
        stored within the attribute outlist but the history of the current
        simulation will be stored within the current_trial attribute.
        
        Parameters
        ----------
            current_s :
                current state
            current_a :
                current action
            reward :
                current reward
            next_s :
                next state
        """
        
        if self.full:
            if self.timestep == 0:
                self.outlist.append([[current_s, \
                            current_a, \
                            reward, \
                            next_s]])
            else:
                self.outlist[self.trial].append([current_s, current_a, \
                    reward, next_s])
        else:
            if (next_s == self.terminal_state):
                self.durations.append(self.timestep)

class MountainCarEnv(rli.Environment):
    """
    Class representing the Mountain Car environment
    
    
    Attributes
    ----------
    
    
    Methods
    -------
    
    """
    def __init__(self, goal = 0.5, rand_start = True, verbose = False):
        
        # Goal state (goal x-position)
        self._goal = goal
        
        self._rand_start = rand_start
        
        # Gravity
        self._g = - 0.0025
        
        self.verbose = verbose
    
    def start_trial(self, *args):
        "Initialize the trial"
        
        if self.rand_start:
            pos = np.random.rand() * (0.5 + 1.2) - 1.2
            vel = np.random.rand() * (0.07 + 0.07) - 0.07 
        else:
            pos = -0.5
            vel = 0.0
        
        return (pos, vel)
    
    def step(self, state, action, t):
        "Step function"
        
        # Update the velocity
        vel_next = state[1] + 0.001*action.move + self.g*np.cos(3.*state[0])
        
        vel = self.bound(vel_next, 'velocity')
        
        # Update the position 
        pos = self.bound(state[0] + vel, 'position')
        
        # Left-hand bound on the position (velocity is set to zero)
        if (pos == -1.2) & (vel < 0):
            vel = 0.0
        #if pos == 0.5:
        #    vel = 0.0
        
        reward = -1.0
        
        next_s = (pos, vel)
        
        # The special value for the terminal state.  
        # Goal state is right-hand position of 0.5
        if pos >= self.goal:
            if self.verbose:
                print("\t goal state reached")
            next_s = self.terminal_state
            reward = 0.0
        
        return reward, next_s
    
    def bound(self, value, deriv):
        
        if deriv == "position":
            if value <= -1.2:
                value = -1.2
            if value >= 0.6:
                value = 0.6
        elif deriv == "velocity":
            if value <= -0.07:
                value = -0.07
            if value >= 0.07:
                value = 0.07
        else:
            if self.verbose:
                print("ERROR, unidentified type")
        
        # state[0] == 0.5 is the goal.  
        
        return(value)
    
    @property
    def g(self):
        "Gravity"
        return self._g
    
    @property
    def rand_start(self):
        "Random starting state"
        return self._rand_start
    
    @property
    def goal(self):
        "Goal state"
        return self._goal
    
    @property
    def terminal_state(self):
        "Placeholder for terminal state"
        return self._terminal_state

class SarsaAgentTraces(rli.Agent):
    """
    """
    def __init__(self, actions, e_start, e_end, e_time2end, alpha, gamma, \
        lamb, tracing = "replace"):
        
        self._epsilon = e_start
        self._epsilon_start = e_start
        self._epsilon_end = e_end
        self._epsilon_time2end = e_time2end
        self._epsilon_incr = float(e_start - e_end)/e_time2end
        
        self._alpha = alpha
        self._gamma = gamma
        self.Q = dict()
        self.visits = dict()
        self.Z = dict()
        self.sa_seen = set()
        
        self._lamb = lamb
        self.tracing = tracing
        
        # List all valid actions
        self._actions = actions
    
    def start_trial(self, state):
        """
        """
        
        # If Q[state] has not been seen before, create an array for it.  
        if not(state in self.Q):
            self.Q[state] = np.zeros(len(self.actions))
            self.visits[state] = 0
        
        # Reset the set of seen states, and the traces
        self.sa_seen = set()
        self.Z = dict()
        self.Z[state] = np.zeros(len(self.actions))
        
        action = utils.egreedy(self.Q[state], self.actions, self.epsilon)
        
        return action
    
    def step(self, s, action, reward, next_s, timestep):
        """
        """
        # Convert action to an index
        action = [i for i, v in enumerate(self.actions) if v == action][0]
        
        # Check for terminal state
        if (next_s == self._terminal_state):
            # Return a dummy action
            next_a = None
            out_action = next_a
        else:
            # Add the current state and action to the set of seen states
            self.sa_seen.add((s,action))
            
            # If Q[s] has not been seen before, create a table for it.
            if not (s in self.Q):
                self.Q[s] = np.empty(len(self.actions))
                self.Q[s][:] = 0.0
                self.visits[s] = 0
            if not (s in self.Z):
                self.Z[s] = np.empty(len(self.actions))
                self.Z[s][:] = 0.0
            
            if not (next_s in self.Q):
                self.Q[next_s] = np.empty(len(self.actions))
                self.Q[next_s][:] = 0.0
                self.visits[next_s] = 0
            if not (next_s in self.Z):
                self.Z[next_s] = np.empty(len(self.actions))
                self.Z[next_s][:] = 0.0
            
            # Find the next action
            next_a = utils.egreedy(self.Q[next_s], self.actions, self.epsilon)
            next_a = [i for i, v in enumerate(self.actions) if v == next_a][0]
            out_action = self.actions[next_a]
            
            delta = reward + self.gamma * self.Q[next_s][next_a] - \
                self.Q[s][action]
            
            # Update Q function and Z traces
            for si, ai in self.sa_seen:
                if self.tracing == "replace":
                    self.Z[si][ai] = np.minimum(self.gamma * self.lamb * self.Z[si][ai], 1)
                else:
                    self.Z[si][ai] = self.gamma * self.lamb * self.Z[si][ai]
            
            # Udpate the traces
            self.Z[s][action] += 1
            
            # Update Q function and Z traces
            for si, ai in self.sa_seen:
                self.Q[si][ai] = self.Q[si][ai] + \
                    self.alpha * delta * self.Z[si][ai]
            
            # Record the observed state
            self.visits[s] += 1
        
        if (timestep == 0) & (self.epsilon > self.epsilon_end):
            self.epsilon -= self.epsilon_incr
        
        return out_action
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value):
        "Allow epsilon to be set, to avoid an AttributeError"
        self._epsilon = value
    
    @property
    def epsilon_start(self):
        "Epsilon starting parameter"
        return self._epsilon_start
    
    @property
    def epsilon_end(self):
        "Epsilon ending parameter"
        return self._epsilon_end
    
    @property
    def epsilon_time2end(self):
        "Number of trials until epsilon reaches the epsilon_end parameter"
        return self._epsilon_time2end
    
    @property
    def epsilon_incr(self):
        "Increment in the epsilon parameter per trial"
        return self._epsilon_incr
    
    @property
    def lamb(self):
        "Lambda parameter"
        return self._lamb
    
    @property
    def gamma(self):
        "Gamma parameter"
        return self._gamma
    
    @property
    def alpha(self):
        "Alpha parameter"
        return self._alpha
    
    @property
    def actions(self):
        "Actions"
        return self._actions


class Move(rli.Action):
    def __init__(self, direction):
        self._move = direction
        self._movedict = {-1: "Reverse", 0: "No throttle", 1: "Forwards"}
        
    def __repr__(self):
        return self._movedict[self.move]
    
    def __str__(self):
        return self._movedict[self.move]
    
    @property
    def move(self):
        "Movement direction"
        return self._move


if __name__ == "__main__":

    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    from matplotlib import pyplot as plt
    
    # Set the random seed
    np.random.seed(100)
    
    car = MountainCarEnv(rand_start = True)
    directions = [-1.0, 0.0, 1.0]
    actions = [Move(x) for x in directions]

    agent = SarsaAgentTraces(actions = actions, e_start = 0.4, \
        e_end = 0.40, e_time2end = 100, alpha = 0.05, gamma = 1.0, lamb = 0.9, \
        tracing = "replace")

    sim = MountainCarSim(agent, car, max_time = np.inf, verbose = True)
    
    import time
    start = time.time()
    N = 9000
    sim.trials(N, max_steps_per_trial = 10000)
    end = time.time() - start
    print(end)
    
    # import am4fmd.plotting as plot
    # color_dict = dict({0:"#f03b20", 1:"#bd0026", 2:"#2ca25f"})
    #
    # a_labels = [str(a) for a in sim.agt.actions]
    # df = utils.dict2df_2d(sim.agt.Q, sim.agt.visits, \
    #     statenames = ['position', 'velocity'])
    # df = df.loc[df.position > -10]
    #
    # fig, ax = plot.plot2dpolicy(df, 'position', 'velocity', 'astar', \
    #     color_dict, "Position", "Velocity", a_labels, visit_thresh = 10)
    #
    # fig, ax = plot.plot2dvaluefn(df, 'position', 'velocity', 'qmax', \
    #      "Position", "Velocity", visit_thresh = 10)
    #
    # fig, ax = plot.plot2dvisits(df, 'position', 'velocity', 'visits', \
    #      "Position", "Velocity", visit_thresh = 10)
    
    p_res = 20; v_res = 20
    
    # We can look at the surface of the cost-to-go function like so...
    x = np.linspace(-1.2, 0.6, num = p_res)
    y = np.linspace(-0.07, 0.07, num = v_res)
    xx, yy, out = utils.dict2array(x, y, sim.agt.Q, p_res, v_res)
    
    # Save to dataframe
    df = pd.DataFrame([xx.flatten(), yy.flatten(), out.flatten()]).T
    df.columns = ["position", "velocity", "cost2go"]
    df.to_csv("results/tables/cost2go.csv", index = False)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    ax.plot_surface(xx, yy, out,
                alpha = 0.9, 
                cmap = plt.cm.viridis,
                edgecolors = "black",
                linewidth = 0.5,
                rstride = 1,
                cstride = 1,
                vmin = 0,
                vmax = 160)
    
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Cost-to-go function\n")
    fig.savefig("results/figures/cost-to-go_function.png", dpi = 300)
    plt.close()
