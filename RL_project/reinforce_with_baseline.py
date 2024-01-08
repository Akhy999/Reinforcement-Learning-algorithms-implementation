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


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, weight_type = 'lecun', hidden_1 = 128):
        super(PolicyNetwork, self).__init__()
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
        x = F.softmax(self.fc2(x), dim=-1)
        x = x.clamp(min=1e-8)
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, input_size, output_size = 1, weight_type = 'lecun', hidden_1 = 128):
        super(ValueNetwork, self).__init__()
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

def episode_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
      R = r + gamma * R
      returns.insert(0, R)
    return torch.Tensor(returns).to(device)

def update_policy_value(states, actions, rewards, last_state, optimizer_policy, optimizer_value, policy, value, gamma = 0.9, lambda_val = 0.95, alpha = 0.01, beta = 0.01):
    returns = episode_returns(rewards, gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    power = torch.arange(0,len(rewards))
    returns = torch.tensor(returns, dtype=torch.float32).to(device)


    log_probs = torch.log(policy(states))
    choosen_action_log_probs = log_probs[range(len(actions)), actions]



    values = value(states)

    value_loss = F.mse_loss(values, torch.reshape(returns, (-1,1)))
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    returns_minus_values = returns - values.view(-1)

    with torch.no_grad():
      returns_minus_values = torch.detach_copy(returns_minus_values)
    gamma_tensor = torch.Tensor(gamma**power).to(device)
    # update wweights
    delta_J = -torch.sum(gamma_tensor * returns_minus_values * choosen_action_log_probs)
    # print(rewards, values_plus_one, values)

    optimizer_policy.zero_grad()
    delta_J.backward()
    optimizer_policy.step()

    return delta_J



from torch.optim import lr_scheduler
episode_states = []
episode_actions = []
episode_rewards = []
episode_scores = []
def reinforce_baseline(env, action_input_size, action_output_size, value_input_size, value_output_size, gamma = 0.9, num_episodes = 1000, lambda_val = 0.95, alpha = 0.01, beta = 0.01, threshold = 200):
    policy = PolicyNetwork(action_input_size, action_output_size).to(device)
    value = ValueNetwork(value_input_size, value_output_size).to(device)

    optimizer_policy = optim.Adam(policy.parameters(), lr=alpha)
    optimizer_value = optim.Adam(value.parameters(), lr=beta)
    scheduler_policy = lr_scheduler.StepLR(optimizer_policy, step_size=20, gamma=0.9)
    scheduler_value = lr_scheduler.StepLR(optimizer_value, step_size=20, gamma=0.9)
    for episode in range(num_episodes):
        scheduler_policy.step()
        scheduler_value.step()
        state = env.reset()
        # state = env.state = [0, 0, 0, 0]

        states = []
        actions = []
        rewards = []
        score = 0
        while True:
          action_probs = policy(torch.tensor(state, dtype=torch.float32).to(device))
          # print(action_probs)
          action = np.random.choice(action_output_size, p=action_probs.cpu().data.numpy())
          next_state, reward, done, _ = env.step(action)

          states.append(state)
          actions.append(action)
          rewards.append(reward)
          score += reward
          state = next_state

          if done:
            pass 
            break
          # env.render()
        episode_states.append(states)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_scores.append(score)
        update_policy_value(states, actions, rewards, next_state, optimizer_policy, optimizer_value, policy, value, gamma, lambda_val, alpha, beta)
        if episode % 100 == 0:
          print(f"Episode: {episode}, Reward: {np.sum(rewards)}")
        #if np.mean(episode_scores[:-100]) > threshold:
        #  print(f"Episode: {episode}, Avg Reward: {np.sum(rewards)}, break")
        #  break


if arg1 == "cartpole":
    reinforce_baseline(env, 4,2 , 4, 1,0.99, 10000, 0.001, 0.001, 0.001, 400)
elif arg1 == "acrobot":
    reinforce_baseline(env, 6,3 , 6, 1,0.99, 10000, 0.001, 0.004, 0.007, 400)

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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.01, 0.01, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.005, 0.01, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.001, 0.01, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.01, 0.1, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.01, 0.2, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.01, 0.3, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.99, 10000, 0.001, 0.4, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.9, 10000, 0.001, 0.4, 0.001, 400)
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
        reinforce_baseline(env, a,b , a, 1,0.8, 10000, 0.001, 0.4, 0.001, 400)
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
