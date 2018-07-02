import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 2500


class Agent:
    def __init__(self, numOfStates, numOfActions):
        self.numOfStates = numOfStates
        self.numOfActions = numOfActions
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.95  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.00007
        self.epsilon_linear_decay = 0.0006
        self.model = self._build_model()
        self.minibatch_size = 20
        self.hiddenLayer1Size = 150
        self.hiddenLayer2Size = 150

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(150, input_dim=self.numOfStates, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.numOfActions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def forget(self, numOfSteps):
        for i in range(numOfSteps):
            self.memory.pop()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.numOfActions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)

    def replay(self):
        # Vectorized method for experience replay
        minibatch = random.sample(self.memory, self.minibatch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.model.predict(np.vstack(minibatch[:, 3]))

            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, \
                                               predict_sprime_target[not_done_indices, \
                                                                     np.argmax(predict_sprime[not_done_indices, :][0],
                                                                               axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(self.minibatch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def decay_epsilon(self, current_episode_num):
        if self.epsilon > self.epsilon_min:
            if (current_episode_num < 400):
                #pass
                self.epsilon -= self.epsilon_linear_decay
            else:
                self.epsilon *= self.epsilon_decay


def getAverageScoreOver100Plays(env, agent):
    total_reward = 0
    for run in range(100):
        state = env.reset()
        state = np.reshape(state, [1, numOfStates])
        while(True):
            action_vals = agent.model.predict(state)
            action = np.argmax(action_vals[0])

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, numOfStates])

            state = next_state
            total_reward += reward

            if done:
                break;

    avgReward =  total_reward/100.0
    return avgReward



if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    numOfStates = env.observation_space.shape[0]
    numOfActions = env.action_space.n
    agent = Agent(numOfStates, numOfActions)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 100

    total_reward = 0

    total_steps = 0

    total_reward_per_episode = []
    greater_than_200_count = 0

    for e in range(EPISODES):
        total_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, numOfStates])

        step_num = 1

        end = False


        while(True):
            total_steps += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, numOfStates])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                total_reward_per_episode.append(total_reward)
                runningAvg= -1000
                if (e > 150):
                    runningAvg = sum(total_reward_per_episode[-150:])/150.0
                    if (runningAvg > 200.0):
                        end = True
                        break;


                print("episode: {}/{}, score: {}, e: {:.2}, reward: {}, runningAvg: {}"
                      .format(e, EPISODES, step_num, agent.epsilon, total_reward, runningAvg))

                break
            if total_steps > agent.minibatch_size:
                agent.replay()

            step_num += 1

        if (end):
            break;

        agent.decay_epsilon(e)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

    avgOver200Plays = getAverageScoreOver100Plays(env, agent)
    print(avgOver200Plays)


