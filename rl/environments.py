import numpy as np
import gym

class RW:
    """
    n-states random walk MRP (Markov Reward Process) 

    - No actions
    - All transitions have 0.5 probability
    - Starting state:
    
    - Non-terminal states: 0,1,...,n-1
    - Terminal states: -1,n
    - Starting state: n//2
    - No actions
    - All transitions have 0.5 probability
    - Reward: always 0 except for the transition in terminal states -1 (-1.0) and n (+1.0)

    args:
        n: number of non-terminal states
        save_current_episode: store the history of the episode
        seed: numpy seed for reproducibility

    """
    
    def __init__(self, n=19, save_current_episode=False, seed=None, terminal_rewards=(-1.0,1.0), reward = 0.0):
        self.rng = np.random.default_rng(seed=seed) # Random generator
        
        self.shape = (n,)
        self.n = n
        self.save_current_episode = save_current_episode
        self.episode = None
        
        self.probs = (0.5,0.5)
        self.terminal_rewards = terminal_rewards
        self.reward = reward
        
        self.done = False # Episode ended
        self.state = None
        self.reset() # Current state
        
        self.only_terminal_rewards = self.reward == 0
            
    def step(self):
        self.state += int(self.rng.choice((-1,1), p=self.probs))
        self.done = self.state in [-1,self.n]
        r = 0.0
        if self.done:
            r = self.terminal_rewards[1] if self.state==self.n else self.terminal_rewards[0]
        self.episode_length += 1
        if self.save_current_episode:
            self.episode.append([r, self.state, None])
        return self.state, r# new_state, reward    

    def reset(self):
        """
        Reset the environment
        """
        self.done = False
        self.episode_length = 0
        self.state = self.n//2
        if self.save_current_episode:
            self.episode = [[None, self.state, None]]
        return self.state
    
    def calculate_value_function(self, gamma):
        if self.reward == 0.0:
            return np.linspace(self.terminal_rewards[0], self.terminal_rewards[1], self.n+2)[1:-1]
        v = np.zeros((self.n,))
        iteration = 0
        updated = True
        while updated and iteration<100000:
            v_new = np.zeros((self.n,))
            for state in range(self.n):
                for i,d in enumerate((-1,1)):
                    new_state = state+d
                    r = 0.0
                    terminal = new_state in [-1,self.n]
                    if terminal:
                        r = self.terminal_rewards[1] if new_state==self.n else self.terminal_rewards[0]
                    else:
                        r = self.reward
                    v_new[state] += self.probs[i]*(r+0.0 if terminal else r+gamma*v[new_state])
            eps = 0.0000001
            if np.linalg.norm(v-v_new) < eps:
                updated = False
            v = v_new
            iteration+=1
        # print(f"Converged in {iteration} iterations")
        return v
    
class GridWorld:
    """
    Simple grid world
    
    - Grid 5xn
    - Terminal state: (4,n-1) 
    - Starting state: (0,0)
    - Actions: down, right, up, left. If with the action the agent exits with probability 0.4 and left or up with probability 0.1 (if I hit a wall, the state doesn't change)
    - Reward: we get -1.0 when we reach one of the states {(1,y) for y in [0,n-2]} or {(3,y) for y in [1,n-1]} and -0.1 otherwise

    args: 
    n: size of the grid world
    save_current_episode: store the history of the episode
    seed: numpy seed for reproducibility
    """
    
    def __init__(self, n=8, save_current_episode=False, step_reward=-0.1, bad_step_reward=-1.0, seed=None):
        self.rng = np.random.default_rng(seed=seed) # Random generator
        
        self.shape = (5,n)
        self.save_current_episode = save_current_episode
        self.episode = None

        # actions
        self.actions = ((1,0),(0,1),(-1,0),(0,-1)) # Actions (down,right,up,left)
        self.num_actions = len(self.actions)
        # rewards
        self.reward = step_reward*np.ones(self.shape)
        for y in range(n-1):
            self.reward[1,y] = bad_step_reward
            self.reward[3,n-1-y] = bad_step_reward
        # policy
        self.p = 0.7# prob to choose right or down
        m = (self.shape[0]/(self.shape[0]+self.shape[1]), self.shape[1]/(self.shape[0]+self.shape[1]))
        self.actions_probs = (self.p*m[0], self.p*m[1], (1.0-self.p)*m[0], (1.0-self.p)*m[1])
        # self.actions_probs = (self.p/2.0, self.p/2.0, (1.0-self.p)/2.0, (1.0-self.p)/2.0)
        
        self.done = False # Episode ended
        self.episode_return = 0.0
        self.episode_length = 0
        self.state = None
        self.reset() # Current state
        
        self.only_terminal_rewards = False
        
    def step(self, a=None):
        """
        Take an action and return the new state and the reward (if no action is provided the action is chosen wrt the distribution self.actions_probs)
        """
        x,y = self.state
        r = 0.0
        if not self.done:
            a = self.rng.choice(self.num_actions, p=self.actions_probs) if a is None else a
            valid = 0 <= x+self.actions[a][0] < self.shape[0] and 0 <= y+self.actions[a][1] < self.shape[1]
            if valid:
                self.state = (x+self.actions[a][0], y+self.actions[a][1])
            r = self.reward[self.state]
            if self.state[0] == self.shape[0]-1 and self.state[1] == self.shape[1]-1:
                self.done = True
        self.episode_return += r
        self.episode_length += 1
        if self.save_current_episode:
            self.episode[-1][-1] = a
            self.episode.append([r, self.state, None])
        return self.state, r# new_state, reward

    def reset(self):
        """
        Reset the environment
        """
        self.done = False
        self.episode_return = 0.0
        self.episode_length = 0
        self.state = (0,0)#(self.rng.integers(self.shape[0]),self.rng.integers(self.shape[1]))
        if self.save_current_episode:
            self.episode = [[None, self.state, None]]
        return self.state
    
    def calculate_value_function(self, gamma):
        v = np.zeros(self.shape)
        iteration = 0
        updated = True
        while updated and iteration<10000:
            v_new = np.zeros(self.shape)
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    if x==self.shape[0]-1 and y==self.shape[1]-1:
                        continue
                    for i,a in enumerate(self.actions):
                        p = self.actions_probs[i]
                        if p>0:
                            valid = -1 < x+a[0] < self.shape[0] and -1 < y+a[1] < self.shape[1]
                            new_x = x+a[0] if valid else x
                            new_y = y+a[1] if valid else y
                            r = self.reward[new_x, new_y]
                            v_new[x,y] += p*(r+gamma*v[new_x,new_y])
            eps = 0.0000001
            if np.linalg.norm(v-v_new) < eps:
                updated = False
            v = v_new
            iteration+=1
        # print(f"Converged in {iteration} iterations")
        return v
    
class GymEnvWrapper:
    """
    Class that wrap environments of the library gym and provide some useful functionalities

    args: 
    env_name: name of the gym environment
    save_current_episode: store a list with the history of the episode
    seed: numpy seed for reproducibility
    """
    
    def __init__(self, env_name, save_current_episode=False, seed=None):
        # Create the environment and set the seed
        self.env_name = env_name
        self.env = gym.make(env_name).env
        self.env.action_space.seed(seed)
        self.env.reset(seed=seed)
        self.rng = self.env.np_random
        
        # Variables with info about the environment
        self.num_actions = self.env.action_space.n
        self.state_limits = [x for x in zip(self.env.observation_space.low, self.env.observation_space.high)] 
        if env_name == 'Acrobot-v1':# If Acrobot, use angles instead cos and sin of the angles
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            self.state_limits = ((-np.pi, np.pi), (-np.pi, np.pi), (low[-2], high[-2]), (low[-1], high[-1]))
        elif 'CartPole' in env_name:
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            limit = 3.0
            # self.state_limits = ((low[-2], high[-2]), (-limit, limit))
            self.state_limits = ((low[0], high[0]), (-limit, limit), (low[-2], high[-2]), (-limit, limit))
        # Other variables
        self.save_current_episode = save_current_episode
        self.episode = None # List of steps in the episode
        self.state = None # Currentstate
        self.done = False # Episode ended
        self.episode_return = 0.0
        self.episode_length = 0
        
        self.only_terminal_rewards = False
        
        # Start the episode
        self.reset()
    
    def is_terminal_state(self, state):
        """
        Check if the state is terminal
        """
        if 'Acrobot' in self.env_name:
            theta1, theta2 = state[0], state[1]
            return bool(-np.cos(theta1) - np.cos(theta2 + theta1) > 1.0)
        elif 'MountainCar' in self.env_name:
            x, v = state[0], state[1]
            return bool(x >= self.env.goal_position and v >= self.env.goal_velocity)
        elif 'CartPole' in self.env_name:
            x, theta = state[0], state[2]
            return bool(x < -self.env.x_threshold or x > self.env.x_threshold or theta < -self.env.theta_threshold_radians or theta > self.env.theta_threshold_radians)
    
    def step(self, a):
        """
        Take the action a and return the new state and the reward
        """
        state, reward, done, truncated, _ = self.env.step(a)
        self.episode_return += reward
        self.episode_length += 1
        self.state = self.env.state
        self.done = done
        if self.save_current_episode:
            self.episode[-1][-1] = a
            self.episode.append([reward, self.state, None])
        return self.state, reward

    def reset(self):
        """
        Reset the environment
        """
        self.done = False
        self.episode_return = 0
        self.episode_length = 0
        self.env.reset()
        self.state = self.env.state
        if 'CartPole' in self.env_name:
            if len(self.state_limits) == 2:
                s = np.array(self.state)
                self.state = (s[2],s[3])
        if self.save_current_episode:
            self.episode = [[None, self.state, None]]
        return self.state
    
def render(env_name, action_value_function, episodes=1, max_steps=500, sec=0.001, seed=None):
    env = GymEnvWrapper(env_name, save_current_episode=False, seed=seed)
    for _ in range(episodes):
        env.reset()
        while not env.done and env.episode_length<max_steps:
            env.env.render()
            action = action_value_function.choose_action()
            env.step(action)
            sleep(sec)
        if env.done:
            print(f"Won in {env.episode_length} steps (total return: {env.episode_return})")
        else:
            print(f"Reached max_steps {env.episode_length} (total return: {env.episode_return})")
    env.env.close()
    
def episode_length_distribution(episodes, environment_class, env_n, seed):
    """
    Function that returns the distribution of episode length

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    episode_length = []
    for episode in range(episodes):
        # Run episode
        env.reset()
        while not env.done:
            state, reward = env.step()
        episode_length.append(env.episode_length)
    return np.array(episode_length)