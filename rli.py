# rli.py
#!/usr/bin/python
"""
Flexible interface for reinforcement learning.  

Reinforcement learning problems require a common set of object.  
This module defines such objects and the place-holder methods for each of 
the objects that will be user-defined. This interface largely follows the code
and C++ template given on Richard Sutton's homepage:  
http://webdocs.cs.ualberta.ca/~sutton/RLinterface/RLI-Cplusplus.html


Classes
-------
Action :
    Represents an action taken by an Agent. Actions are applied to the 
    Environment class and output of an Agent class.  
State :
    Represents a state of an Environment. Output by Environment objects and are
    input to Agent objects.  
Agent :
    Represents a decision-maker in the system. Takes input as State objects
    and generates Action objects.  
Environment : 
    Represents the environment in which the learn task is to take place. 
    Action objects are applied to Environment and States are output.  
Simulation : 
    Class representing a simulation. Manages the interaction between
    Environment and Agent objects including saving data, displaying 
    information, and ending simulations.  

Author: W. Probert 2014.  
"""
import time, copy

class Action(object):
    """
    Represents an action.  
    
    Objects of class Action are output from the Agent class and input to the 
    Environment class.  
    """
    def __init__(self):
        pass

class State(object):
    """
    Represents a state of the environment.  
    
    State objects are generated by Environments and input to Agents.  
    """
    def __init__(self):
        pass

class Agent(object):
    """
    Class representing the decision-making agent of the system.  
    
    Users need to define start_trial and step functions.  Public attributes 
    will likely be the hyperparameters associated with the RL algorithm.  
    
    Methods:
        start_trial: Initialize the Agent for the start of a trial
        step:  step the agent through one timestep of the simulation
    """
    
    def __init__(self):
        pass
    
    def start_trial(self, s):
        """Initialize the Agent for the start of a trial.  
        
        This function is user-defined and will generally output the first 
        action in the trial to be applied in the environment.  This function
        is called at the start of each new trial.  The first state, s, is 
        output by the environment.  
        
        Args:
            s: starting state of the environment
        
        Returns:
            The first action of the agent.  
        """
        return Action()
    
    def step(self, s, a, r, next_s, timestep):
        """Step the Agent through one timestep of the simulation.  
        
        This function is user-defined and represents the main functionality
        of the agent.  That is, this is where learning, if any, occurs.  Any 
        updates to value-functions or policies will occur within this step 
        function, including updates using function approximation techniques, or
        methods using eligibility traces.  If the next state is the terminal 
        state then it may be useful to return a dummy action.  
        
        Args:
            s: current state of the environment
            a: action taken in the current state
            r: observed reward from taking action a in state s
            next_s: observed next state of the environment
        
        Returns:
            The next action to be taken given the next state.  
        """
        
        return Action()

class Environment(object):
    """
    Class representing the Environment in a reinforcement learning task.  
    
    Users need to define the start_trial and step methods, provide a starting
    state and define a goal state.  
    
    Methods:
        start_trial: prepare the environment for the state of a trial.  
        step: step the environment through one timestep of the simulation.  
    
    """
    def __init__(self):
        self._start = None
        self._goal = None
        
    def start_trial(self):
        """Initialize the environment for the start of a trial
        
        This function initializes the environment for the start of a trial.  
        This is user-define and generally involves simply returning the 
        starting state.  
        
        Returns:
            The starting state of the environment.  
        """
        
        return self.start
    
    def step(self, s, a, t):
        """Step the environment through one timestep of the simulation
        
        User-defined step function for the environment.  This function defines
        how an environment changes state through one time step in the 
        simulation given action a is taken.  This function is called by the 
        Simulation object in the RL framework.  This function returns both the 
        next state of the environment and the immediate reward.  If the 
        terminal state is reached it may be desirable to return a special state
        such as None or 0.  
        
        Args:
            s: the current state of the environment
            a: the action taken by the agent in the current state
        
        Returns:
            The reward given transition to the next state and the next state.  
        """
        
        reward = None
        next_s = None
        
        return reward, next_s
    
    @property
    def start(self):
        "Returns the starting states"
        return self._start

    @property
    def goal(self):
        "Returns the goal state"
        return self._goal


class Simulation(object):
    """
    Class representing the simulation in a reinforcement learning task.  
    
    This object manages the interaction between agent and environment, collects 
    data, and manages the display, if any. Users will define the collect_data 
    and the full2obs methods.  The start_trial, steps, and trials methods are 
    not expected to be changed by users.  
    
    Attributes
    ----------
        agt: Agent
            Agent to be used in the simulation.  
        
        env: Environment
            Environment to be used in the simulation.  
            
        max_time: float
            maximum walltime (in hrs) for all trials or timesteps to run.
            The simulation stops if the max_time is reached.  
        
        verbose: boolean
            Logical to trigger printing-to-screen of simulation progress
        
        full : boolean
            Logical to trigger saving of all states, rewards and actions
    
    Methods
    -------
        start_trial: 
            initialize a trial
            
        steps: 
            run the simulation through a specified number of time steps
            
        trials: 
            run the simulation through a specified number of trials
            
        full2obs: 
            translate from full state to observed state
            
        collect_data: 
            save any data at each step of the simulation
        
    """
    def __init__(self, Agent, Environment, max_time = 1000, \
        verbose = False, full = False):
        """
        Initialize the simulation object
        
        Save Agent and Environment objects and call the start_trial method so 
        the simulation is ready for the steps() or trials() methods.  
        Instantiation of a simulation object sets the current state and action
        because of the call to the start_trial method.  The terminal state 
        needs to be set by the user.  Users may wish to define extra attributes
        of the simulation to store data.  
        """
        
        self.agt = Agent
        self.env = Environment
        self.max_time = max_time
        self.verbose = verbose
        self.full = full
        
        self.current_s = None
        self.current_a = None
        
        self._terminal_state = "terminal"
        self.durations = []
        self.total_culls = []
        self.infected_premises = []
        self.outlist = []
        
        self.agt._terminal_state = self._terminal_state
        self.env._terminal_state = self._terminal_state
        
        self.start_trial()
    
    def start_trial(self):
        """
        Start a reinforcement learning trial (ie episode).  
        
        The function calls the start_trial method of both the Environment and 
        Agent in the simulation which respectively return the first Sensation
        (ie state) and Action to be taken in the simulation.  This function 
        also resets the timestep counter to zero.  Custom methods may compute 
        particular statistics per trial and/or update any relevant graphics.
        """
        if self.verbose:
            print('starting simulation')
        # Set the current time step
        self.timestep = 0
        
        if self.verbose:
            print('starting environment')
        
        # Find the first state
        self.current_s = self.env.start_trial()
        
        # Find the first action given the current state
        observed_s = self.full2obs(self.current_s)
        
        if self.verbose:
            print('starting agent')
        self.current_a = self.agt.start_trial(observed_s)
        
    def steps(self, num_steps):
        """
        Run the simulation for a specific number of timesteps.  
        
        The simulation is run for num_steps timesteps from the current state of
        the environment, looping through the step methods of both the 
        environment and the agent until the terminal state is reached.  If the
        number of time steps to be run is longer than the time it takes to 
        reach the terminal state, then the simulation continues until the 
        specified number of timesteps is reached.  For each new trial that 
        is started the simulation calls the start_trial method of the 
        simulation object.  Note that the change in state from terminal state
        a new starting state is not counted as a timestep.  
        
        Args:
            num_steps: number of time steps for which to run the simulation.  
        """
        
        if (self.current_s == self.terminal_state):
            self.start_trial()
        
        self.trial = 0
        for ts in range(num_steps):
            
            # Step the environment to find a new state
            reward, next_s = self.env.step(self.current_s, \
                self.current_a, t = self.timestep)
            
            # Save the data
            self.collect_data(self.current_s, self.current_a, reward, next_s)
            
            # Find the next action
            observed_s = self.full2obs(self.current_s)
            observed_next_s = self.full2obs(next_s)
            next_a = self.agt.step(observed_s, self.current_a, reward, observed_next_s, ts)
            
            # Check if next state is the terminal state.
            if (not next_s.terminal): # != self.terminal_state):
                self.current_a = next_a
                self.current_s = next_s
                self.timestep += 1
            else:
                self.start_trial()
                self.trial += 1
                self.timestep = 0
    
    def trials(self, num_trials, max_steps_per_trial = 10000):
        """
        Run the simulation for num_trials trials (aka episodes)
        
        This function runs the simulation for num_trials trials, starting from 
        whatever state the environment is in. Trial are limited to be less than
        max_steps_per_trial steps. Each trial is initialized with start_trial 
        and is completed when the terminal state is reached (or when 
        max_steps_per_trial is reached).  
        
        Args:
            num_trials: number of trials for which to run the simulation
            max_steps_per_trial: maximum number of steps witin any trial
        """
        if (self.current_s == self.terminal_state):
            self.start_trial()
        
        # Find the current time
        start_time = time.time()
        
        # Step through using the environment's function
        for self.trial in range(num_trials):
            if ((time.time() - start_time) >= self.max_time*60*60):
                break
            
            self.timestep = 0
            if self.verbose:
                print("Trial num:", self.trial)
            
            while( (self.timestep < max_steps_per_trial) & \
                (self.current_s != self.terminal_state)):
                
                # Step the environment to find a new state
                r, next_s = self.env.step(self.current_s, self.current_a, t = self.timestep)
                # this could have outputs of r, next_s, terminal (boolean)
                
                # Save the data
                self.collect_data(self.current_s, self.current_a, r, next_s)
                
                # Find the next action (from the observed state)
                observed_s = self.full2obs(self.current_s)
                observed_next_s = self.full2obs(next_s)
                
                next_a = self.agt.step(observed_s, self.current_a, r, observed_next_s, self.timestep)
                
                # Reset state and action
                self.current_a = next_a
                self.current_s = next_s
                self.timestep += 1
            
            if not (self.timestep < max_steps_per_trial):
                if self.verbose:
                    print("Max steps per trial reached")
            
            if self.verbose:
                print("Time steps:", self.timestep)
            
            # Restart the next trial (unless it's the last trial)
            if (self.trial != (num_trials-1)):
                self.start_trial()
    
    def full2obs(self, s):
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
        return s
    
    def collect_data(self, s, a, r, next_s):
        """Collect any data at each time step of the simulation.  
        
        This function is called once at each step of the simulation. By
        default this method does not save any data.  Users may customise what 
        data are saved by the simulation by modifying this function.  
        
        Args:
            s: current state of the environment
            a: action taken in the current state
            r: reward observed from taking action a in state s
            next_s: resultant next state from taking action a in state s
        """
        
        # save list of all action, states, rewards if desired
        if self.full:
            if self.timestep == 0:
                self.outlist.append([[\
                    copy.deepcopy(s), \
                    copy.deepcopy(a), \
                    copy.deepcopy(r), \
                    copy.deepcopy(next_s)\
                    ]])
            else:
                self.outlist[self.trial].append([\
                    copy.deepcopy(s), \
                    copy.deepcopy(a), \
                    copy.deepcopy(r), \
                    copy.deepcopy(next_s)\
                    ])
        
        if (next_s.terminal):
            culls = sum(s.n_cattle_original[next_s.status < 0])
            self.total_culls.append(culls)
            self.durations.append(self.timestep)
            self.infected_premises.append(sum(s.status == -1))
    
    @property
    def terminal_state(self):
       "Return the terminal state"
       return self._terminal_state
