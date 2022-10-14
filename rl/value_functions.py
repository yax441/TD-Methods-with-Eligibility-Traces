from math import prod,floor
import numpy as np
from rl.tile_coding import IHT,tiles,get_tiling_size

class StateValueFunction:
    """
    State value function (with optional state aggregation)
    
    args:
        env: environment
        initial_value: initial value of weights
        gamma: discount factor
        aggr_groups: number of groups for state aggregation
    """
    def __init__(self, env, initial_value=0.0, gamma=1.0, aggr_groups = None):
        self.env = env
        self.aggr_groups = aggr_groups
        self.shape = (aggr_groups,)*len(env.shape) if aggr_groups else env.shape
        self.true_value_function = env.calculate_value_function(gamma)
        self.initial_value = initial_value
        self.w = None # values to learn
        self.updates = np.zeros(self.shape) # saved updates (single update strategy)
        self.reset_weights()
        
    def reset_weights(self):
        """
        Reset the weights to initial_value
        """
        self.w = np.full(self.shape, self.initial_value)
    
    def get_index(self, state):
        """
        Get the index of the state
        """
        if isinstance(state, int):
            return int(self.aggr_groups*state/self.env.shape[0])  if self.aggr_groups else state
        return tuple(int(self.aggr_groups*s/d) for s,d in zip(tuple(state), self.env.shape)) if self.aggr_groups else state
    
    def states_are_equal(self, state1, state2):
        if isinstance(state1, int):
            return state1 == state2
        for x,y in zip(self.get_index(state1), self.get_index(state2)):
            if x != y:
                return False
        return True
        
    def get_value(self, state):
        """
        Get the value of the approximated action-value function in (state,a)
        """
        return self.w[self.get_index(state)]
    
    def get_value_by_index(self, index):
        """
        Get the value of the approximated action-value function in (state,a)
        """
        return self.w[index]
    
    def update_saved(self):
        """
        Update and reset saved updates
        """
        self.w += self.updates
        self.updates = np.zeros(self.shape)
    
    def update_state(self, state, d):
        """
        Update state value with d
        """
        self.w[self.get_index(state)] += d
    
    def update(self, d):
        """
        Update value function with d
        """
        self.w += d
    
    def save_update_state(self, state, d):
        """
        Save update d for state
        """
        self.updates[self.get_index(state)] += d
    
    def save_update(self, d):
        """
        Update value function with d
        """
        self.updates += d
    
    def calculate_return(self, t, n, T, gamma_powers):
        """
        Calculate the return G_t:t+n with discount factor gamma
        """
        G = 0.0
        if self.env.only_terminal_rewards:#avoid for loop if rewards are given only in terminal states
            i = min(t+n,len(self.env.episode)-1)
            G += gamma_powers[i-(t+1)]*self.env.episode[i][0]
        else:
            for i in range(t+1,min(t+n+1,len(self.env.episode))):
                G += gamma_powers[i-(t+1)]*self.env.episode[i][0]
        if t+n<T:
            r, s, a = self.env.episode[t+n]
            G += gamma_powers[n]*self.w[self.get_index(s)]
        return G
    
    def calculate_λreturn(self, t, T, lambda_powers, gamma_powers):
        """
        Calculate the return G_t:t+n with discount factor gamma
        """
        G = 0.0
        for n in range(1, T - t):
            G += lambda_powers[n-1]*self.calculate_return(t, n, T, gamma_powers) # lambda^(n-1)*G_{t,n}
        G *= 1.0-lambda_powers[1]
        G += lambda_powers[T-t-1]*self.calculate_return(t, T-t, T, gamma_powers)
        return G
    
    def calculate_rmse(self):
        if self.aggr_groups:
            err = 0.0
            for index in np.ndindex(self.env.shape):
                err += (self.get_value(index) - self.true_value_function[index])**2
            return np.sqrt(err/prod(self.env.shape))
        return np.sqrt(np.mean((self.w - self.true_value_function)**2))
    
    def calculate_rmse_v(self, v):
        if self.aggr_groups:
            err = 0.0
            for index in np.ndindex(self.env.shape):
                err += (v[self.get_index(index)] - self.true_value_function[index])**2
            return np.sqrt(err/prod(self.env.shape))
        return np.sqrt(np.mean((v - self.true_value_function)**2))

    def min_rmse(self):
        if self.aggr_groups:
            v = np.zeros(self.shape)
            count = np.zeros(self.shape)
            for index in np.ndindex(self.env.shape):
                i = self.get_index(index)
                v[i] += self.true_value_function[index]
                count[i] += 1.0
            return self.calculate_rmse_v(v/count)
        return 0.0

def value_function_info(EnvironmentClass, n_env, gamma, aggr_groups):
    """
    Returns the true value function and the minimum rmse possible

    args:
        EnvironmentClass: class of the environment
        n_env: size of the environment
        gamma: discount factor
        aggr_groups: number of groups for state aggregation
    """
    env = EnvironmentClass(n=n_env)
    value_function = ValueFunction(env, 0.0, gamma, aggr_groups)
    print("True value function: ", value_function.true_value_function)
    print("Minimum rmse:", value_function.min_rmse())
    
class ActionValueFunction:
    """
    Action value function (with optional state aggregation)
    
    args:
        env: environment
        initial_value: initial value of weights
        aggr_groups: number of groups in which to divide each dimension
    """
    def __init__(self, env, initial_value=0.0, aggr_groups = None):
        self.env = env
        self.initial_value = initial_value
        self.aggr_groups = aggr_groups
        self.shape = (aggr_groups,)*len(env.shape) if aggr_groups else env.shape
        self.w = None # values to learn
        self.shape = (*self.shape, self.env.num_actions)
        self.updates = np.zeros(self.shape) # saved updates (single update strategy)
        self.reset_weights()
        
    def reset_weights(self):
        """
        Reset the weights to initial_value
        """
        self.w = np.full(self.shape, self.initial_value)
    
    def get_index(self, state, action):
        """
        Get the index of (state,action)
        """
        if isinstance(state, int):
            return int(self.aggr_groups*state/self.env.shape[0])  if self.aggr_groups else (state,action)
        return (*tuple(int(self.aggr_groups*s/d) for s,d in zip(tuple(state), self.env.shape)),action) if self.aggr_groups else (*state,action)
    
    def get_value(self, state, action):
        """
        Get the value of the approximated action-value function in (state,a)
        """
        return self.w[self.get_index(state,action)]
    
    def get_value_by_index(self, index):
        """
        Get the value of the approximated action-value function in index
        """
        return self.w[index]
    
    def update_saved(self):
        """
        Update and reset saved updates
        """
        self.w += self.updates
        self.updates = np.zeros(self.shape)
        
    def update_state(self, state, action, d):
        """
        Update state value with d
        """
        self.w[self.get_index(state,action)] += d
    
    def update(self, d):
        """
        Update value function with d
        """
        self.w += d
    
    def save_update_state(self, state, action, d):
        """
        Save update d for state
        """
        self.updates[self.get_index(state,action)] += d
    
    def save_update(self, d):
        """
        Update value function with d
        """
        self.updates += d
    
    def choose_action(self, epsilon, state=None):
        """
        Choose an action following an epsilon-greedy policy
        """
        a = None
        if self.env.rng.binomial(1, epsilon) == 1:
            a = self.env.rng.integers(0,self.env.num_actions)
        else:
            if state is None:
                state = self.env.state
            values = [self.get_value(state, a) for a in range(self.env.num_actions)]
            max_value = max(values)
            a = self.env.rng.choice([i for i, val in enumerate(values) if val == max_value])
        if self.env.save_current_episode:
            self.env.episode[-1][-1] = a
        return a

    def calculate_return(self, t, n, T, gamma_powers):
        """
        Calculate the return G_t:t+n with discount factor gamma
        """
        G = 0.0
        for i in range(t+1,min(t+n+1,len(self.env.episode))):
            G += gamma_powers[i-(t+1)]*self.env.episode[i][0]
        if t+n<T:
            r, s, a = self.env.episode[t+n]
            G += gamma_powers[n]*self.w[self.get_index(s,a)]
        return G

class ActionValueFunctionTiling:
    """
    Action value function with function approximation (hash tiling)
    
    args:
    env: environment
    initial_value: initial value of weights
    num_tilings: 
    """
    def __init__(self, env, initial_value=0.0, num_tilings=8, initial_size=4096):
        self.env = env
        self.max_size = get_tiling_size(env,num_tilings)
        initial_size = min(self.max_size, initial_size)
        self.initial_size = initial_size
        self.shape = (initial_size,)
        self.num_tilings = num_tilings
        self.initial_value = initial_value
        self.w = None # weights to learn
        self.reset_weights()
        self.updates = np.zeros((initial_size,))
        self.iht=IHT(self.max_size) # hash table for tiling
        self.scale = [self.num_tilings/abs(high-low) for low,high in env.state_limits]
        
    def reset_weights(self):
        """
        Reset the weights to initial_value
        """
        self.w = np.full((self.initial_size,), self.initial_value)
    
    def pad(self, new_size):
        # print('pad',new_size)
        self.shape = (new_size,)
        self.w = np.pad(self.w, (0,new_size+1-self.w.size), constant_values=self.initial_value)
        self.updates = np.pad(self.updates, (0,new_size+1-self.updates.size))
        
    def get_index(self, state, a):
        """
        Get the num_tilings active indexes for the pair (state,a)
        """
        if self.env.is_terminal_state(state):
            return []
        index = tiles(self.iht,self.num_tilings,[state[i]*scale for i,scale in enumerate(self.scale)],[a])
        # pad arrays if needed
        max_index = max(index)
        if max_index>self.w.size-1:
            self.pad(max_index)
        return index
    
    def get_value(self, state, action):
        """
        Get the value of the approximated action-value function in (state,a)
        """
        index = self.get_index(state,action)
        return np.sum(self.w[index])
    
    def get_value_by_index(self, indexes):
        """
        Get the sum of elements of the weights vector with index in the list indexes provided
        """
        return np.sum(self.w[indexes])
    
    def update_saved(self):
        """
        Update and reset saved updates
        """
        self.w += self.updates
        self.updates = np.zeros(self.shape)
        
    def update_state(self, state, action, d):
        """
        Update state value with d
        """
        index = self.get_index(state,action)
        self.w[index] += d
    
    def update_by_index(self, index, d):
        """
        Update state value with d
        """
        self.w[index] += d
    
    def update(self, d):
        """
        Update value function with d
        """
        self.w += d
    
    def save_update_state(self, state, action, d):
        """
        Save update d for state
        """
        index = self.get_index(state,action)
        self.updates[index] += d
    
    def save_update_by_index(self, index, d):
        """
        Save update d for state
        """
        self.updates[index] += d
        
    def save_update(self, d):
        """
        Update value function with d
        """
        self.updates += d
    
    def choose_action(self, epsilon, state=None):
        """
        Choose an action following an epsilon-greedy policy
        """
        # if self.env.done:
        #     return 0
        if self.env.rng.binomial(1, epsilon) == 1:
            return self.env.rng.integers(0,self.env.num_actions)
        if state is None:
            state = self.env.state
        indexes = [self.get_index(state, a) for a in range(self.env.num_actions)]
        values = [self.get_value_by_index(index) for index in indexes]
        max_value = max(values)
        a = self.env.rng.choice([i for i, val in enumerate(values) if val == max_value])
        if self.env.save_current_episode:
            self.env.episode[-1][-1] = a
        return a
    
    def calculate_return(self, t, n, T, gamma_powers):
        """
        Calculate the return G_t:t+n with discount factor gamma
        """
        G = 0.0
        for i in range(t+1,min(t+n+1,len(self.env.episode))):
            G += gamma_powers[i-(t+1)]*self.env.episode[i][0]
        if t+n<T:
            r, s, a = self.env.episode[t+n]
            G += gamma_powers[n]*self.get_value(s, a)
        return G