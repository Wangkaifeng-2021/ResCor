import torch
import torch.nn as nn
import torch.nn.functional as F


def datatype_transmission(states, device):
    features = torch.as_tensor(states[0], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=device)
    mask = torch.as_tensor(states[2], dtype=torch.float32, device=device)

    return features, adjacency, mask


class SafetyActorModel(nn.Module):
    def __init__(self, N, F, A):
        super(SafetyActorModel, self).__init__()
        self.num_agents = N
        self.input_dim = F + A
        self.hidden_dim = 128
        self.output_dim = A

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pi = nn.Linear(self.hidden_dim, self.output_dim)

        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, observation, action_probs, actions_mask=None):
        X_in, _, RL_indice = datatype_transmission(observation, self.device)
        action = torch.as_tensor(action_probs, dtype=torch.float32, device=self.device)

        x = torch.cat((X_in, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        corrected_probs = self.pi(x)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))
        corrected_probs = torch.mul(corrected_probs, mask)

        if actions_mask is not None:
            corrected_probs = corrected_probs + (actions_mask * 1e10 - 1e10)

        return corrected_probs


class SafetyCriticModel(nn.Module):
    def __init__(self, N, F, A):
        super(SafetyCriticModel, self).__init__()
        self.num_agents = N
        self.input_dim = F + A
        self.hidden_dim = 128
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.output_dim)

        self.device = torch.device("cpu")

        self.to(self.device)

    def forward(self, observation, action_probs):
        X_in, _, RL_indice = datatype_transmission(observation, self.device)
        action = torch.as_tensor(action_probs, dtype=torch.float32, device=self.device)

        x = torch.cat((X_in, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        corrected_value = self.value(x)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))
        corrected_value = torch.mul(corrected_value, mask)

        return corrected_value
