import gym
import numpy as np
import REINFORCE
import matplotlib.pyplot as plt

gamma = 0.99

def main():

    env = gym.make('CartPole-v0')
    model = REINFORCE.Reinforce()
    history = {'rewards': [], 'episode': []}

    for i_episode in range(1500):
        state = env.reset()
        total_reward = 0
        rewards = []
        for t in range(1, 200):  # Don't infinite loop while learning
            env.render()
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)


            total_reward = gamma*total_reward + reward
            rewards.append(total_reward)

            if done:
                history['rewards'].append(t + 1)
                history['episode'].append(i_episode)
                print("Episode finished after {} timesteps".format(t + 1))
                break

        model.update(total_reward, rewards)

    plt.plot(history['episode'], history['rewards'], 'orange')
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
