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


import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
        # weight_funcion(self.fc1.weight.data)
        
        self.fc2 = nn.Linear(hidden_1, output_size)
        # weight_funcion(self.fc2.weight.data)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from torch.optim import lr_scheduler
import random
episode_states = []
episode_actions = []
episode_rewards = []
episode_scores = []
def SemiGradSarsaEpisodic(n, env, action_input_size, action_output_size, gamma = 0.9, num_episodes = 1000, alpha = 0.01, epsilon = 0.25, epsilon_end = 0.05, threshold = 200):
    stateActionValue = StateActionValueNetwork(action_input_size, action_output_size)
    stateActionValue.to(device)
    optimizer_state_action_value = optim.Adam(stateActionValue.parameters(), lr=alpha)
    scheduler_state_action_value = lr_scheduler.StepLR(optimizer_state_action_value, step_size=25, gamma=0.9)
    
    torch.autograd.set_detect_anomaly(True)
    for episode in range(num_episodes):
        scheduler_state_action_value.step()
        state = env.reset()
        states = []
        actions = []
        rewards = []
        S = []
        A = []
        R = []
        Q = []
        score = 0
        # print("epsilon:",epsilon)
        step_count = 0

        action_vals = stateActionValue(torch.tensor(state, dtype=torch.float32).to(device))
        max_action = torch.argmax(action_vals)
        epsilon_by_n = epsilon / action_output_size
        action_probs = np.zeros(( action_output_size)) + epsilon_by_n
        action_probs[max_action] += 1 - epsilon
        
        action = np.random.choice(action_output_size, p=action_probs)
        # print(action_vals[action])
        S.append(state)
        R.append(1)
        rewards.append(1)
        A.append(action)
        Q.append(action_vals[action])
        T = float('inf')
        t = 0
        while True:
          # print("-",t%(n+1))
          step_count += 1
          # epsilon *= 0.9999
          if t<T:
            next_state, reward, done, _ = env.step(A[t])
            # print("t",t)
            R.append(reward)
            S.append(next_state)
            states.append(next_state)
            rewards.append(reward)
            score += reward
            if done:
              T = (t+1)
            else:
              next_action_vals = stateActionValue(torch.tensor(S[t+1], dtype=torch.float32).to(device))
              next_max_action = torch.argmax(next_action_vals)
              epsilon_by_n = epsilon / action_output_size
              next_action_probs = np.zeros(( action_output_size)) + epsilon_by_n
              next_action_probs[next_max_action] += 1 - epsilon
              # print(action_probs)
              next_action = np.random.choice(action_output_size, p=next_action_probs)
              A.append(next_action)
              # if Q[(t+1)%(n+1)] == 0:
              Q.append(next_action_vals[next_action])
              # print(next_action)
              actions.append(next_action)
          # print("T", t - n + 1)
          T_small = t - n + 1
          if T_small >= 0:
            # print("hi")
            G = 0
            # print(T_small+n, T)
            for x in range(T_small+1, min(T_small+n, T)+1):
              G += (gamma ** (x-T_small-1)) * R[x]
            # print("hi2")
            if T_small + n < T :
              q_T_small_n = stateActionValue(torch.tensor(S[T_small+n], dtype=torch.float32).to(device))[A[T_small+n]]
              G = G + (gamma ** n)*q_T_small_n#Q[(T_small+n)]
            G = torch.tensor(G, dtype=torch.float32).to(device)
            # with torch.no_grad():
              # G = torch.detach_copy(G)
            # print("loss")
            # print(T_small, A, A[T_small], S[T_small])
            q_T_small = stateActionValue(torch.tensor(S[T_small], dtype=torch.float32).to(device))[A[T_small]]
            loss = nn.functional.mse_loss(G, q_T_small)
            optimizer_state_action_value.zero_grad()
            loss.backward()
            optimizer_state_action_value.step()
            # if done :
              # print("loss:", loss.item())
          if T_small==T-1:
            break

          # states.append(state)
          # actions.append(action)
          # rewards.append(reward)
          # score += reward
          # state = next_state

          
          t += 1
              

        epsilon = max(epsilon_end, epsilon * 0.995)
        episode_states.append(states)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_scores.append(score)
        # update_policy_value(states, actions, rewards, next_state, optimizer_policy, optimizer_value, policy, value, gamma, lambda_val, alpha, beta)
        if episode % 1 == 0:
          print(f"Episode: {episode}, Avg Reward: {np.sum(rewards)}")
        if np.mean(episode_scores[:-100]) > threshold and np.std(episode_scores[:-100]) < 50:
          print(f"Episode: {episode}, Avg Reward: {np.sum(rewards)}, break")
          break
    print(states)
    return states, actions, rewards



if arg1 == "cartpole":
    SemiGradSarsaEpisodic(10, env, 4, 2, 1, 500, 0.01, 0.5, 0.001, 200)
elif arg1 == "acrobot":
    SemiGradSarsaEpisodic(10, env, 6, 3, 1, 500, 0.01, 0.5, 0.001, 200)

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
elif arg1 == "cartpole":
    s = 4
    a = 2
