import gymnasium as gym
from torch.nn import functional as F
from itertools import count
from typing_extensions import Self
import math
import random
import torch as T
from torch import nn, optim

MEMORY_BATCH: int = 128
GAMMA: float = 0.99
TUR: float = 0.005
LR: float = 1e-4
EPSILON_START: float = 0.9
EPSILON_END: float = 0.05
EPSILON_ENTROPY: float = 1e-3

class DQN(nn.Module):
    def __init__(self: Self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(n_observations, 128)
        self.fc2: nn.Linear = nn.Linear(128, 128)
        self.fc3: nn.Linear = nn.Linear(128, n_actions)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class Memory:
    def __init__(self: Self, max_length: int, device: T.device) -> None:
        self.data: T.Tensor = T.zeros((max_length, 10), device=device, dtype=T.float32)
        self.index: int = 0
        self.is_full: bool = False

    def append(self: Self, *elements: tuple[T.Tensor]) -> None:
        if self.index >= len(self.data):
            self.index = 0
            self.is_full = True
        self.data[self.index] = T.tensor(elements)
        self.index += 1

    def __len__(self: Self) -> int:
        return len(self.data) if self.is_full else self.index

    def sample(self: Self, k: int) -> T.Tensor:
        return self.data[T.randint(len(self), (k,))]

env: gym.Env = None
device: T.device = None
policy_net: DQN = None
target_net: DQN = None
optimizer: optim.Adam = None
criterion: nn.SmoothL1Loss = None
memory: Memory = None
iteration: int = None

def optimization_handler():
   
    if len(memory) < MEMORY_BATCH:
        return

    batch: T.Tensor = memory.sample(MEMORY_BATCH)
    state_batch = batch[:, :4]
    action_batch = batch[:, 4].to(T.int64).view(-1, 1)
    next_state_batch = batch[:, 5:9]
    reward_batch = batch[:, 9]

    non_final_mask: T.Tensor = ~T.all(next_state_batch == 0, dim=1)
    non_final_next_states = next_state_batch[non_final_mask]

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = T.zeros(MEMORY_BATCH, device=device)
    with T.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = criterion(state_action_values, expected_state_action_values.view(-1, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    nn.utils.clip_grad_value_(policy_net.parameters(), 100)


def select_action(state: T.Tensor, iteration: int) -> int:
    epsilon_threshold: float = epsilon(iteration)
    if random.random() > epsilon_threshold:
        with T.no_grad():
            return T.argmax(policy_net(state)).item()
    return env.action_space.sample()


def epsilon(x: int) -> float:
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1 * EPSILON_ENTROPY * x)


def train_model(epochs: int) -> None:
 
    for epoch in range(epochs):
        state, info = env.reset()
        state = T.tensor(state, device=device)
        for t in count():
          
            # Итерации эпохи
            global iteration
            iteration += 1
            action: int = select_action(state, iteration)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = T.tensor(reward, device=device)
            done = terminated or truncated
            env.render()

            memory.append(*state, action, *([0] * state.numel() if terminated else observation), reward,)
            state = T.tensor(observation, device=device)

            # Оптимизация модели
            optimization_handler()

            target_dict: dict = target_net.state_dict()
            policy_dict: dict = policy_net.state_dict()

            assert target_dict.keys() == policy_dict.keys()
            for key in target_dict:
                target_dict[key] = policy_dict[key] * TUR + target_dict[key] * (1 - TUR)
            target_net.load_state_dict(target_dict)           
            if done:
                break

        print(f"Эпоха {epoch} - количество итераций на удержание фиговины ({t})")

def main() -> None:

    global env, iteration, memory, device, policy_net, target_net, optimizer, criterion
    
    env = gym.make("CartPole-v1", render_mode="human")
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), LR)

    
    criterion = nn.SmoothL1Loss()
    memory = Memory(10_000, device=device)
    iteration = 0    
    train_model(400)

if __name__ == "__main__":
    main()