import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)


arg1 = sys.argv[1]

if arg1 == "acrobot":
    env = gym.make('Acrobot-v1' )
    print("Environment: Acrobot-v1")
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
elif arg1 == "cartpole":
    env = gym.make('CartPole-v1' )
    print("Environment: CartPole-v1")
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)


class StateActionValueNetwork(nn.Module):
    def __init__(self, input_size, output_size, weight_type='lecun', hidden_1=128, hidden_2=256):
        super(StateActionValueNetwork, self).__init__()

        if weight_type == 'lecun':
            weight_funcion = nn.init.xavier_uniform_
        elif weight_type == 'kaiming':
            weight_funcion = nn.init.kaiming_uniform_
        elif weight_type == 'random_uniform':
            weight_funcion = nn.init.uniform_

        self.fc1 = nn.Linear(input_size, hidden_1)
        weight_funcion(self.fc1.weight.data)
        
        self.fc2 = nn.Linear(hidden_1, output_size)
        weight_funcion(self.fc2.weight.data)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def update_state_action_value_dq(action, action_vals, state, next_state, reward, gamma, optimizer_state_action_value, stateActionValue, offPolicy, done, batch):

    state, action, reward, next_state, done = zip(*batch)

    state = torch.tensor(state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)
    reward = torch.tensor(reward, dtype=torch.int64).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.int64).to(device)

    q_values = stateActionValue(state).gather(1, action.unsqueeze(1))
    next_q_values = offPolicy(next_state).max(1)[0].detach()
    target_values = reward + (1 - done) * gamma * next_q_values

    state_action_value_loss = nn.functional.mse_loss(q_values, target_values.unsqueeze(1))
    optimizer_state_action_value.zero_grad()
    state_action_value_loss.backward()
    optimizer_state_action_value.step()

    return state_action_value_loss


episode_states = []
episode_actions = []
episode_rewards = []
episode_scores = []
def DeepQLearn(env, action_input_size, action_output_size, gamma = 0.9, num_episodes = 1000, alpha = 0.01, epsilon = 0.25, epsilon_end = 0.05, threshold = 200):
    stateActionValue = StateActionValueNetwork(action_input_size, action_output_size).to(device)
    offPolicy = StateActionValueNetwork(action_input_size, action_output_size).to(device)
    offPolicy.load_state_dict(stateActionValue.state_dict())
    offPolicy.eval() # set eval mode to mae sure it doesn't train

    optimizer_state_action_value = optim.Adam(stateActionValue.parameters(), lr=alpha)
    scheduler_state_action_value = lr_scheduler.StepLR(optimizer_state_action_value, step_size=25, gamma=0.9)
    off_policy_memory = []
    for episode in range(num_episodes):
        # epsilon *= 0.999
        scheduler_state_action_value.step()
        state = env.reset()
        states = []
        actions = []
        rewards = []
        score = 0
        ##print("epsilon:",epsilon)
        step_count = 0
        while True:
          step_count += 1
          # epsilon *= 0.9999
          action_vals = stateActionValue(torch.tensor(state, dtype=torch.float32).to(device))
          max_action = torch.argmax(action_vals)
          # print(action_vals, max_action)
          epsilon_by_n = epsilon / action_output_size
          action_probs = np.zeros(( action_output_size)) + epsilon_by_n
          action_probs[max_action] += 1 - epsilon
          # print(action_probs)
          action = np.random.choice(action_output_size, p=action_probs)
          next_state, reward, done, _ = env.step(action)
          off_policy_memory.append((state, action, reward, next_state, done))
          if len(off_policy_memory) > 32:
            batch = random.sample(off_policy_memory, 32)
            # update_q_network(q_network, target_network, optimizer, batch)

            loss = update_state_action_value_dq(action, action_vals, state, next_state, reward, gamma, optimizer_state_action_value, stateActionValue, stateActionValue, done, batch)


          states.append(state)
          actions.append(action)
          rewards.append(reward)
          score += reward
          state = next_state

          if done :
            #print("loss:", loss.item())
            pass ## update policy
            break

        if episode % 10 ==0:
            offPolicy.load_state_dict(stateActionValue.state_dict())
        epsilon = max(epsilon_end, epsilon * 0.995)
        episode_states.append(states)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_scores.append(score)
        if episode % 100 == 0:
          print(f"Episode: {episode}, Reward: {np.sum(rewards)}")
        if np.mean(episode_scores[:-100]) > threshold:
         print(f"Episode: {episode}, Reward: {np.sum(rewards)}, break")
         break
    return states, actions, rewards, stateActionValue

if arg1 == "cartpole":
    DeepQLearn(env, 4, 2, 1, 500, 0.01, 0.1, 0.001, 400)
elif arg1 == "acrobot":
    DeepQLearn(env, 6, 3, 1, 500, 0.001, 0.4, 0.001, -100)

## Moving average window = 5
d = []
sum = 0
for i in range(len(episode_scores)-5):
  sum += episode_scores[i]
  d.append(np.mean(episode_scores[i:i+5]))
plt.plot(d)
plt.xlabel("Number of Episodes")
plt.ylabel("Performance")
plt.title("Moving Average with indow = 5")
# plt.plot(episode_scores[1001:])

## Best curve comparision
# plt.plot(s[:700])
# plt.plot(rb[:700])
# plt.plot(d[:700])
# plt.legend(['Semi-Gradient N step SARSA', 'Reinforce with Baseline', 'Deep Q-Learning'])
# plt.title("Acrobot - Best curves for all 3 algorithms ")
# plt.xlabel("Number of Episodes")
# plt.ylabel("Performance")

#alpha
if arg1 == "acrobot":
    s = 6
    a = 3

if False:
    print("alpha=",0.01)
    episode_scores_10_avg_01 = [0 for i in range(500)]
    episode_scores_10_avg_01_all = []
    for i in range(5):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 1, 500, 0.01, 0.1, 0.001, 200)
        for j in range(500):
            episode_scores_10_avg_01[j] += episode_scores[j]/5
        episode_scores_10_avg_01_all.append(episode_scores)

    print("alpha=",0.005)
    episode_scores_10_avg_005 = [0 for i in range(500)]
    episode_scores_10_avg_005_all = []
    for i in range(5):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 1, 500, 0.005, 0.1, 0.001, 200)
        for j in range(500):
            episode_scores_10_avg_005[j] += episode_scores[j]/5
        episode_scores_10_avg_005_all.append(episode_scores)
        
    print("alpha=",0.001)
    episode_scores_10_avg_001 = [0 for i in range(500)]
    episode_scores_10_avg_001_all = []
    for i in range(5):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 1, 500, 0.001, 0.1, 0.001, 200)
        for j in range(500):
            episode_scores_10_avg_001[j] += episode_scores[j]/5
        episode_scores_10_avg_001_all.append(episode_scores)

    plt.plot(episode_scores_10_avg_01)
    plt.plot(episode_scores_10_avg_005)
    plt.plot(episode_scores_10_avg_001)
    plt.xlabel("Number of Episodes")
    plt.xlabel("Performance")
    plt.title(f"Learning Curve with varied alpha and epsilon={0.01}, gamma={0.99}")
    plt.legend(['alpha = 0.01', 'alpha = 0.005', 'alpha = 0.001'])

    #beta
    print("epsilon=",0.1)
    episode_scores_10_avg_01 = [0 for i in range(500)]
    episode_scores_10_avg_01_allb = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        DeepQLearn(env, a, b, 1, 500, 0.001, 0.1, 0.001, 450) ## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_01[j] += episode_scores[j]/10
        episode_scores_10_avg_01_allb.append(episode_scores)

    print("epsilon=",0.2)
    episode_scores_10_avg_001 = [0 for i in range(500)]
    episode_scores_10_avg_001_allb = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        DeepQLearn(env, a, b, 1, 500, 0.001, 0.2, 0.001, 450) ## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_001[j] += episode_scores[j]/10
        episode_scores_10_avg_001_allb.append(episode_scores)

    print("epsilon=",0.3)
    episode_scores_10_avg_005 = [0 for i in range(500)]
    episode_scores_10_avg_005_allb = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 1, 500, 0.001, 0.3, 0.001, 450)## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_005[j] += episode_scores[j]/10
        episode_scores_10_avg_005_allb.append(episode_scores)

    plt.plot(episode_scores_10_avg_01)

    plt.plot(episode_scores_10_avg_001)
    plt.plot(episode_scores_10_avg_005)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Performance")
    plt.title(f"Learning Curve with varied epsilon and alpha={0.01}, gamma={0.99}")
    plt.legend(['epsilon = 0.1', 'epsilon = 0.2', 'epsilon = 0.3'])

    print("gamma=",0.99)
    episode_scores_10_avg_01 = [0 for i in range(500)]
    episode_scores_10_avg_01_allg = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 0.99, 500, 0.001, 0.4, 0.001, 450) ## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_01[j] += episode_scores[j]/10
        episode_scores_10_avg_01_allg.append(episode_scores)

    print("gamma=",0.9)
    episode_scores_10_avg_001 = [0 for i in range(500)]
    episode_scores_10_avg_001_allg = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 0.9, 500, 0.001, 0.4, 0.001, 450) ## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_001[j] += episode_scores[j]/10
        episode_scores_10_avg_001_allg.append(episode_scores)

    print("gamma=",0.8)
    episode_scores_10_avg_005 = [0 for i in range(500)]
    episode_scores_10_avg_005_allg = []
    for i in range(10):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_scores = []
        #alpha
        DeepQLearn(env, a, b, 0.8, 500, 0.001, 0.4, 0.001, 450)## 0.004, 0.007 acrobat ## 0.01 0.01
        for j in range(500):
            episode_scores_10_avg_005[j] += episode_scores[j]/10
        episode_scores_10_avg_005_allg.append(episode_scores)

    plt.plot(episode_scores_10_avg_01)

    plt.plot(episode_scores_10_avg_001)
    plt.plot(episode_scores_10_avg_005)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Performance")
    plt.title(f"Learning Curve with varied gamma and alpha={0.001}, epsilon={0.4}")
    plt.legend(['gamma = 0.99', 'gamma = 0.9', 'gamma = 0.8'])

    plt.plot(episode_scores_10_avg_01)

    # plt.plot(episode_scores_10_avg_001)
    # plt.plot(episode_scores_10_avg_005)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Performance")
    plt.title(f"Best Learning Curve alpha={0.001}, epsilon={0.4}, gamma={0.99}")
    # plt.legend(['gamma = 0.99', 'gamma = 0.9', 'gamma = 0.8'])
