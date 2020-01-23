import gym
import numpy as np
import REINFORCE
import actor_critic
from PPO import PPO
import matplotlib.pyplot as plt

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = True
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 1000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    hidden_layer = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    mode = 'PPO'
    #############################################

    memory = Memory()
    if mode == 'R':
        model = REINFORCE.Reinforce(state_dim, action_dim, hidden_layer, lr, betas, gamma)
    elif mode == 'A2C':
        model = actor_critic.A2C(state_dim, action_dim, hidden_layer, lr, betas, gamma)
    elif mode == 'PPO':
        model = PPO(state_dim, action_dim, hidden_layer, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    history = {'rewards': [], 'episode': []}
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            if mode == 'PPO':
                action = model.policy_old.act(state, memory)
            else:
                action = model.act(state)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if mode == 'PPO':
                if timestep % update_timestep == 0:
                    model.update(memory)
                    memory.clear_memory()
                    timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        if mode != 'PPO':
            model.update(memory)
            memory.clear_memory()

        avg_length += t

        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval * solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
        #     break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            history['rewards'].append(avg_length)
            history['episode'].append(i_episode)

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    plt.plot(history['episode'], history['rewards'], 'orange')
    plt.xlabel('Episodes')
    plt.ylabel('Avg-Rewards')
    plt.show()
    env.close()


# def main():
#     env = gym.make('CartPole-v0')
#     # print(env.observation_space.shape[0])
#
#
#     if mode == 'R':
#         model = REINFORCE.Reinforce()
#     elif mode == 'A2C':
#         model = actor_critic.A2C()
#     history = {'rewards': [], 'episode': []}
#
#     for i_episode in range(1500):
#         state = env.reset()
#         total_reward = 0
#         rewards = []
#         for t in range(1, 200):  # Don't infinite loop while learning
#             env.render()
#             action = model.select_action(state)
#             state, reward, done, _ = env.step(action)
#
#
#             total_reward = gamma*total_reward + reward
#             rewards.append(total_reward)
#
#             if done:
#                 history['rewards'].append(t + 1)
#                 history['episode'].append(i_episode)
#                 print("The {}th episode finished after {} timesteps".format(i_episode, t + 1))
#                 break
#
#         model.update(total_reward, rewards)
#



if __name__ == '__main__':
    main()
