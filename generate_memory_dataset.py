import gym
import gym_pathfinding
import numpy as np
import operator
import itertools
import json
from tqdm import tqdm

from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from gym_pathfinding.games.astar import astar
from gym_pathfinding.envs.partially_observable_env import partial_grid

from memory_pathfinding import MemoryPathfindingEnvConfig, get_show_function_from_spec

class DatasetGenerator():
    """See MemoryPathfindingEnvConfig docstring"""

    def __init__(self, shape, grid_type, observable_depth, show_function, timesteps):
        self.shape = shape
        self.grid_type = grid_type
        self.observable_depth = observable_depth
        self.show_function = show_function
        self.timesteps = timesteps

    def generate_dataset(self, size):
        """
        Return
        ------
        return episodes, a list of tuple (images, labels)

        each episode contains a list of :
        image : (m, n, 2) grid with state and goal on the 3rd axis
            state = (m, n) grid with 1 (wall), 0 (free) and -1 (unseen) ;
            goal = (m, n) grid with 10 at goal position
        label : the action made
        """
        episodes = []
        for _ in tqdm(range(size)):
            grid, start, goal = generate_grid(self.shape, grid_type=self.grid_type)
            path, action_planning = compute_action_planning(grid, start, goal)

            episode = self.generate_episode(grid, goal, action_planning, path)

            images, labels = zip(*episode)

            episodes.append((images, labels))
        return episodes

    def generate_episode(self, grid, goal, action_planning, path):
        visible_goal_grid = create_goal_grid(grid.shape, goal)
        invisible_goal_grid = np.zeros(grid.shape, dtype=np.int8)

        for timestep in range(self.timesteps):
            # at the end, pad the episode with the last action
            if (timestep < len(action_planning)): 
                action = action_planning[timestep]
                position = path[timestep]

                # Compute the partial grid
                _partial_grid = partial_grid(grid, position, self.observable_depth)
                _partial_grid = grid_with_start(_partial_grid, position)

                # Goal grid contains something only if the goal is visible
                goal_grid = visible_goal_grid if _partial_grid[goal] != -1 else invisible_goal_grid
                
                # Stack partial and goal grid
                image = np.stack([_partial_grid, goal_grid], axis=2)
            
            if (self.show_function(timestep)):
                image = np.stack([grid_with_start(grid, position), visible_goal_grid], axis=2)

            yield image, action

# reversed MOUVEMENT dict
ACTION = {mouvement: action for action, mouvement in dict(enumerate(MOUVEMENT)).items()}

def compute_action_planning(grid, start, goal):
    path = astar(grid, start, goal)

    action_planning = []
    for i in range(len(path) - 1):
        pos = path[i]
        next_pos = path[i+1]
        
        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        mouvement = tuple(map(operator.sub, next_pos, pos))

        action_planning.append(ACTION[mouvement])
        
    return path, action_planning


def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid

def grid_with_start(grid, start_position):
    _grid = np.array(grid, copy=True)
    _grid[start_position] = 2
    return _grid



def main():
    import joblib
    import argparse

    parser = argparse.ArgumentParser(description='Generate data, list of (images, labels)')
    parser.add_argument("--env_spec", type=str, default="./env_spec.json", help=MemoryPathfindingEnvConfig.__doc__)
    parser.add_argument('--size', '-s', type=int, default=10000, help='Number of example')
    parser.add_argument('--out', '-o', type=str, default='./data/dataset.pkl', help='Path to save the dataset')
    args = parser.parse_args()

    spec = spec_from_json(MemoryPathfindingEnvConfig, args.env_spec)


    generator = DatasetGenerator(
        shape=(spec.height, spec.width),
        grid_type=spec.grid_type, 
        observable_depth=spec.obs_depth,
        show_function=get_show_function_from_spec(spec),
        timesteps=spec.seq_length
    )

    dataset = generator.generate_dataset(args.size)

    print("Saving data into {}".format(args.out))
    joblib.dump(dataset, args.out)
    print("Done")

def spec_from_json(SpecClass, jsonfile):
    return SpecClass(**json.load(open(jsonfile)))

if __name__ == "__main__":
    main()
