import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

class DeepQLWithER:
    def __init__(self, gym_environment):
        self.env = gym_environment
        self.numOfStates = gym_environment.observation_space.shape[0]
        self.numOfActions = gym_environment.action_space.n

        self.numOfEpisodes = 2500

        self.replayMemoryCapacity = 10000
        self.replayMemory = deque(maxlen = self.replayMemoryCapacity)

        self.gamma = 0.99

        self.epsilon = 0.95
        self.epsilon_minimum = 0.01
        self.epsilon_exponential_decay = 0.96
        self.epsilon_linear_decay = 0.0006

        self.minibatch_size = 20

        #Neural Net stuff
        self.hiddenLayer1Size = 150
        self.hiddenLayer2Size = 150
        self.nn_learning_rate = 0.00007

        #Build the Keras NN Model
        self.nn_model = Sequential()
        self.nn_model.add(Dense(self.hiddenLayer1Size, input_dim=self.numOfStates, activation='relu'))
        self.nn_model.add(Dense(self.hiddenLayer2Size, activation='relu'))
        self.nn_model.add(Dense(self.numOfActions, activation='linear'))
        self.nn_model.compile(loss='mse', optimizer=Adam(lr=self.nn_learning_rate))


    def select_action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.numOfActions)
        return np.argmax(self.nn_model.predict(state)[0])

    def replay_memory_with_miniBatch(self):
        if (len(self.replayMemory) > self.minibatch_size):
            batch = np.array(random.sample(self.replayMemory, self.minibatch_size))
            newYs = []
            for state, action, reward, next_state, terminal_state in batch:
                if (terminal_state):
                    y = reward
                else:
                    y = reward + self.gamma * np.amax(self.nn_model.predict(next_state)[0])
                model_y = self.nn_model.predict(state)
                model_y[0][action] = y
                newYs.append(model_y)

            states = np.vstack(batch[:, 0])
            newYs = np.vstack(newYs)
            self.nn_model.fit(states, newYs, epochs=1, verbose=0)

    def decay_epsilon(self, current_episode_num):
        if self.epsilon > self.epsilon_minimum:
            if (current_episode_num < 400):
                self.epsilon -= self.epsilon_linear_decay
            else:
                self.epsilon *= self.epsilon_exponential_decay


    def getAverageScoreOver100Plays(self):
        total_reward = 0
        for run in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.numOfStates])
            while(True):
                action_values = self.nn_model.predict(state)
                action = np.argmax(action_values[0])

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.numOfStates])

                state = next_state
                total_reward += reward

                print("run_num: {}, reward: {}".format(run, reward))

                if done:
                    break;

        avgReward =  total_reward/100.0
        return avgReward

    def runAlgorithm(self):
        total_steps = 0
        total_reward_per_episode = []

        for episode_num in range(self.numOfEpisodes):
            total_reward = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.numOfStates])
            step_num = 1
            end = False

            while (True):
                total_steps += 1
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.numOfStates])

                #store transition in D
                self.replayMemory.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward

                if done:
                    total_reward_per_episode.append(total_reward)
                    runningAvg = None
                    if (episode_num > 150):
                        runningAvg = sum(total_reward_per_episode[-150:]) / 150.0
                        if (runningAvg > 200.0):
                            end = True
                            break;

                    print("episode_num: {}, NumOfGameSteps: {}, epsilon: {:.4}, reward: {}, runningAvg150Episodes: {}"
                          .format(episode_num, step_num, self.epsilon, total_reward, runningAvg))

                    break

                self.replay_memory_with_miniBatch()
                step_num += 1
            if (end):
                break;

            self.decay_epsilon(episode_num)

        self.nn_model.save_weights("KerasWightsFile")

        avgOver100Plays = self.getAverageScoreOver100Plays()
        print ("Done training. Average score over 100 plays using the trained agent .." + avgOver100Plays)

if __name__ == "__main__":
    DQL = DeepQLWithER( gym.make('LunarLander-v2'))
    DQL.runAlgorithm()

