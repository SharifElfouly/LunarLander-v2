"""
Training process:
    1) Run X number of random ship landings (Initial population)
    2) Average over the rewards of all X ships
    3) Take the best Y percent of ships and train a NN (One generation)
    4) NN decides actions of the next gen with a little randomness (exploration)
    5) Reset NN and train with the best Y percent of ships
    6) Jump to 4)
"""
import gym
import numpy as np
import random
from sklearn.preprocessing import Normalizer

# to save np arrays.
from tempfile import TemporaryFile

from nn import NN

GAME = "LunarLander-v2"
env = gym.make(GAME)

def run_env_randomly(n_runs, n_frames=100, render=False):
    """Runs the game environment with a random agent.

    Saves all observations and actions of a random agent and its sum of
    rewards at the end of the environment.

    Args:
        n_runs (int): Number of times the environment is run.
        render (bool, optional): Should the environment be displayed graphicly.
            Default False.
        n_frames (int, optional): Max limit of frames. Defaults to 100.

    Returns:
        ships_infos (list): One row in the list represents the infos of one ship
            as follows [id, sum_rewards, ship_observations, ship_actions].
    """
    ships_infos = []

    id = 0

    for i in range(n_runs):

        if i % 50 == 0:
            print(i)

        observation = env.reset()
        done = False

        ship_reward = [] # reward of one ship
        ship_observations = [] # observation of one ship
        ship_actions = [] # actions of one ship

        while not done:

            prev_observation = []

            for _ in range(n_frames):
                if render:
                    env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                if len(prev_observation) > 0:

                    ship_reward.append(reward)
                    ship_observations.append(prev_observation)
                    ship_actions.append(action)

                prev_observation = observation

        sum_rewards = sum(ship_reward)

        ship_infos = [id, sum_rewards, ship_observations, ship_actions]
        ships_infos.append(ship_infos)

        id += 1

    return ships_infos

def run_env_with_nn(n_runs, nn, normalizer, exploration_rate, n_frames=100):
    """"Runs the environment with the NN taking decitions.

    Args:
        n_runs (int): Number of times the environment is run.
        nn (keras.models): NN that will make the decisions.
        exploration_rate (float): Determines how much randomness (exploration)
            will take place. For example if exploration_rate is 0.1, 10% of all
            actions will be random.
        n_frames (int, optional): Max limit of frames. Defaults to 100.
        normalizer (sklearn normalizer object): The normalizer used for
            training. New observations have to be normalized.

    Returns:
        ships_infos (list): One row in the list represents the infos of one ship
            as follows [id, sum_rewards, ship_observations, ship_actions].
    """
    ships_infos = []

    id = 0

    for _ in range(n_runs):

        observation = env.reset()
        done = False

        ship_reward = [] # reward of one ship
        ship_observations = [] # observation of one ship
        ship_actions = [] # actions of one ship

        while not done:

            prev_observation = []

            for _ in range(n_frames):

                # normalize observation.
                observation = normalizer.transform(observation.reshape(8, -1))

                prev_observation = observation

                # if true take random action.
                if exploration_rate > random.uniform(0, 1):
                    action = env.action_space.sample()
                else:
                action = np.argmax(nn.prediction(observation))

                observation, reward, done, info = env.step(action)

                if len(prev_observation) > 0:
                    ship_reward.append(reward)
                    ship_observations.append(prev_observation)
                    ship_actions.append(action)

        sum_rewards = sum(ship_reward)

        ship_infos = [id, sum_rewards, ship_observations, ship_actions]
        ships_infos.append(ship_infos)

        id += 1

    return ships_infos


def sort_out_ships(rejection_rate, ships_infos, verbose=True):
    """Sorts out bad ships.

    Looks at all the ships and rejects the worst ones based on the
    rejection_rate. If the rejection_rate is 0.4 for example, the best 60% of
    ships are selected. Returned are the good actions and observations.

    Args:
        rejection_rate (float): Rejection rate between (0, 1)
        ships_infos (list): One row in the list represents the infos of one ship
            as follows [id, sum_rewards, ship_observations, ship_actions].
        verbose (bool, optional): If true the rewards of the best 3 ships are
            printed out. Defaults to true.

    Returns:
        good_observations_actions_pairs (list): These are the good observations
            and corresponding actions of the good ships.
    """
    n_ships = len(ships_infos)
    n_good_ships = int(n_ships * (1 - rejection_rate))

    sorted_ships_infos = sorted(ships_infos, key=lambda x: x[1], reverse=True) # sort by reward

    # print out the rewards of the 3 best ships.
    if verbose:
        for i in range(3):
            print(sorted_ships_infos[i][1])

    good_observations_actions_pairs = []

    for i in range(n_good_ships):
        good_observations = sorted_ships_infos[i][2]
        good_action = sorted_ships_infos[i][3]
        good_observations_actions_pairs.append([good_observations, good_action])

    return good_observations_actions_pairs


def random_observation_action_pairs(observations_actions_pairs):
    """Returns a list of a observation and its action.

    At the end the list is shuffled.

    Args:
        observations_actions_pairs (list): Observations and actions of multiple
            ships.

    Return:
        observation_action_pairs (np.array): list of a observation and its
            corresponding action.
    """
    observation_action_pairs = []

    for i in range(len(observations_actions_pairs)):
        for j in range(len(observations_actions_pairs[i][0])):
            observation = observations_actions_pairs[i][0][j]
            action = observations_actions_pairs[i][1][j]

            observation_action_pairs.append([list(observation), action])

    observation_action_pairs = random.sample(observation_action_pairs, len(observation_action_pairs))

    return np.array(observation_action_pairs)


def normalize(X):
    """Normalizes data.

    Args:
        X (np.array): States (Shape=(None, 8, 1)).

    Returns:
        X (np.array): Normalized X.
        X_normalizer (sklearn normalizer object): Normalizer object.
    """
    X_normalizer = Normalizer()
    X = X_normalizer.fit_transform(X)

    return X, X_normalizer

def average_rewards(ships_infos):
    """Averages the reward of one generation.

    Args:
        ships_infos (list): One row in the list represents the infos of one ship
            as follows [id, sum_rewards, ship_observations, ship_actions].

    Returns:
        average_return (float): Average return of one generation.
    """

    rewards = []

    for ship in ships_infos:
        reward = ship[1]
        rewards.append(reward)

    average_return = sum(rewards) / float(len(rewards))
    return average_return


if __name__ == '__main__':
    SAVE_INITIAL_POPULATION = True
    RANDOM_SHIPS = 10000
    REJECTION_RATE_RANDOM_POPULATION = 0.95
    EPOCHS = 15
    GENERATIONS = 100

    REJECTION_RATE = 0.95
    POPULATION_SIZE = 2000
    EXPLORATION_RATE = 0.12

    RENDER = False

    random_ships = run_env_randomly(RANDOM_SHIPS, render=RENDER)
    good_ships = sort_out_ships(REJECTION_RATE_RANDOM_POPULATION, random_ships)
    observation_action_pairs = random_observation_action_pairs(good_ships)

    if SAVE_INITIAL_POPULATION:
        outfile = TemporaryFile()
        np.save(outfile, observation_action_pairs)

    print('Training set size: {}'. format(str(len(observation_action_pairs))))

    # just a bad way to create a clean numpy array.
    obs = observation_action_pairs[:, 0]
    observations = []
    for observation in obs:
        observations.append(list(observation))

    observations = np.array(observations).reshape(len(observations), 8)
    # noramlize and save the scaler object.
    norm_observations, normalizer = normalize(observations)

    actions = observation_action_pairs[:, 1]
    # complicated way to create one hot vectors
    one_hot_actions = np.zeros((len(actions), 4))
    one_hot_actions[np.arange(len(actions)), list(actions)] = 1
    one_hot_actions = one_hot_actions.reshape(len(actions), 4)

    # generation number 0 of random ships.
    generation = 0

    navigator = NN(epochs=EPOCHS)
    navigator.train(norm_observations, one_hot_actions)
    navigator.save('generation_{}'.format(generation))

    average_reward = average_rewards(random_ships)
    print('Random generation 0: {}'.format(average_reward))

    for _ in range(GENERATIONS):

        generation += 1

        navigated_ships = run_env_with_nn(POPULATION_SIZE, navigator, normalizer, EXPLORATION_RATE)
        good_ships = sort_out_ships(REJECTION_RATE, navigated_ships)
        observation_action_pairs = random_observation_action_pairs(good_ships)

        print('Training set size: {}'. format(str(len(observation_action_pairs))))

        # just a bad way to create a clean numpy array.
        obs = observation_action_pairs[:, 0]
        observations = []
        for observation in obs:
            observations.append(list(observation))

        observations = np.array(observations).reshape(len(observations), 8)
        # noramlize and save the scaler object.
        norm_observations, normalizer = normalize(observations)

        actions = observation_action_pairs[:, 1]
        # complicated way to create one hot vectors
        one_hot_actions = np.zeros((len(actions), 4))
        one_hot_actions[np.arange(len(actions)), list(actions)] = 1
        one_hot_actions = one_hot_actions.reshape(len(actions), 4)

        navigator = NN(epochs=EPOCHS)
        navigator.train(norm_observations, one_hot_actions)
        navigator.save('generation_{}'.format(generation))

        average_reward = average_rewards(navigated_ships)
        print('Generation {}: {}'.format(generation, average_reward))
