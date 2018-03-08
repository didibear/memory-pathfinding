import gym
from gym_pathfinding.envs.pathfinding_env import PathFindingEnv
from gym_pathfinding.envs.partially_observable_env import partial_grid
import random

class MemoryPathfinding(gym.Env):
    """
    Parameter
    ---------

    show_state: a function that return weither or not to show the grid state (boolean)
        def show_state(timestep)
    """

    def __init__(self, lines, columns, observable_depth, *, 
            grid_type="free",  
            show_state=lambda timestep: False, # = total_hide
            screen_size=(640, 640)
            ):

        self.env = PathFindingEnv(lines, columns, 
            grid_type=grid_type, 
            screen_size=screen_size
        )
        self.observable_depth = observable_depth
        self.show_state = show_state

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.timestep = 0

    def reset(self):
        self.timestep = 0
        state = self.env.reset()
        
        if not self.show_state(self.timestep):
            return self.partial_state(state)
        
        return state

    def step(self, action):
        self.timestep += 1
        state, reward, done, info = self.env.step(action)

        if not self.show_state(self.timestep):
            state = self.partial_state(state)

        return state, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def render(self, mode='human'):
        grid = self.env.game.get_state()
        
        if not self.show_state(self.timestep):
            grid = self.partial_state(grid)

        if (mode == 'human'):
            self.env.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.env.close()

    def partial_state(self, state):
        return partial_grid(state, self.env.game.player, self.observable_depth)



# Show function :

def total_show(timestep):
    return True

def total_hide(timestep):
    return False

def show_sometimes(show_probability, seed):
    rng = random.Random(seed)
    def show_state(timestep):
        return rng.random() <= show_probability
    return show_state

def show_start(nb_steps):
    def show_state(timestep):
        return timestep < nb_steps
    return show_state










