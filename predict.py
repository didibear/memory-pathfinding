import numpy as np
import tensorflow as tf
import time
import json
from collections import namedtuple

from memory_pathfinding import MemoryPathfindingEnv, MemoryPathfindingEnvConfig
from macn.model import MACN, MACNConfig


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("env_spec", "./env_spec.json", MemoryPathfindingEnvConfig.__doc__)
tf.flags.DEFINE_string("model_spec", "./model_spec.json", MACNConfig.__doc__)

tf.flags.DEFINE_integer("batch_size",   32,  "Batch size (batch of episode)")

# Model
tf.flags.DEFINE_string('weights', "./model/weights.ckpt", "File to load the model weights")

# Tests
tf.flags.DEFINE_integer("episodes",     100,   "Number of episodes to test")
tf.flags.DEFINE_integer("test_seed",    1,      "The seed to generate test environment")

tf.flags.DEFINE_boolean("render", True, "Weither or not to show the env grid")

def main(args):
    env_spec = spec_from_json(MemoryPathfindingEnvConfig, FLAGS.env_spec)
    model_spec = spec_from_json(MACNConfig, FLAGS.model_spec)
    
    env = MemoryPathfindingEnv.from_spec(env_spec)
    macn = MACN.from_spec(model_spec)
    
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, FLAGS.weights)

        dones = 0
        for episode in range(FLAGS.episodes):
            env.seed(FLAGS.test_seed * episode)
            print(episode, end="\r")

            model_state = sess.run([macn.state_in])

            state = env.reset()
            for timestep in range(15):
                if FLAGS.render :
                    env.render()
                    time.sleep(0.2)

                grid, grid_goal = parse_state(state)

                actions_probabilities, model_state = sess.run([macn.prob_actions, macn.state_out], feed_dict={
                    macn.X: [np.stack([grid, grid_goal], axis=2)],
                    macn.state_in: model_state
                })
                
                action = np.argmax(actions_probabilities)
                state, reward, done, _ = env.step(action)

                if done:
                    dones += 1
                    break
        print("Accuracy : {:.02f} %".format(100 * dones / FLAGS.episodes))
        
        env.close()


def parse_state(state):
    goal = state == 3
    state[goal] = 0

    return state, create_goal_grid(state.shape, goal)

def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid


def spec_from_json(SpecClass, jsonfile):
    return SpecClass(**json.load(open(jsonfile)))

if __name__ == "__main__":
    tf.app.run()