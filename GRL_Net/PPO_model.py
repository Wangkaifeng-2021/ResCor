import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, SAGEConv
from torch_geometric.utils import dense_to_sparse


def datatype_transmission(states, device):
    features = torch.as_tensor(states[0], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=device)
    mask = torch.as_tensor(states[2], dtype=torch.float32, device=device)

    return features, adjacency, mask


# ------Graph Actor Model------ #
class Graph_Actor_Model(nn.Module):
    def __init__(self, N, F, A, gnn):
        super(Graph_Actor_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.gnn = gnn

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # GNN
        if self.gnn == 'GCN':
            # self.GCN = GCNConv(128, 128)
            self.GCN = SGConv(256, 256, K=2, add_self_loops=False)
            self.Graph_Dense = nn.Linear(256, 256)
        elif self.gnn == 'GAT':
            heads = 4
            self.GAT_1 = GATConv(128, 128, heads=heads, add_self_loops=False)
            self.GAT_2 = GATConv(128 * heads, 128, heads=heads, add_self_loops=False)
            self.Graph_Dense = nn.Linear(128 * heads, 128)
        elif self.gnn == 'GraphSAGE':
            self.SAGE_1 = SAGEConv(128, 128)
            self.SAGE_2 = SAGEConv(128, 128)
            self.Graph_Dense = nn.Linear(128, 128)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        # policy
        self.policy_1 = nn.Linear(512, 128)
        self.policy_2 = nn.Linear(128, 128)

        # actor network
        self.pi = nn.Linear(128, A)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, actions_mask=None):

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GNN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        if self.gnn == 'GCN':
            X_graph = self.GCN(X, A_in_Sparse)
            X_graph = F.relu(X_graph)
            X_graph = self.Graph_Dense(X_graph)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GAT':
            X_graph_1 = self.GAT_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.GAT_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GraphSAGE':
            X_graph_1 = self.SAGE_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.SAGE_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        F_concat = torch.cat((X_graph, X), 1)

        # policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        pi = self.pi(X_policy)
        pi = torch.mul(pi, mask)

        if actions_mask is not None:
            pi = pi + (actions_mask * 1e10 - 1e10)

        return pi


# ------Graph Critic Model------ #
class Graph_Critic_Model(nn.Module):
    def __init__(self, N, F, A, gnn):
        super(Graph_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.gnn = gnn

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # GNN
        if self.gnn == 'GCN':
            # self.GCN = GCNConv(128, 128)
            self.GCN = SGConv(256, 256, K=2, add_self_loops=False)
            self.Graph_Dense = nn.Linear(256, 256)
        elif self.gnn == 'GAT':
            heads = 4
            self.GAT_1 = GATConv(128, 128, heads=heads, add_self_loops=False)
            self.GAT_2 = GATConv(128 * heads, 128, heads=heads, add_self_loops=False)
            self.Graph_Dense = nn.Linear(128 * heads, 128)
        elif self.gnn == 'GraphSAGE':
            self.SAGE_1 = SAGEConv(128, 128)
            self.SAGE_2 = SAGEConv(128, 128)
            self.Graph_Dense = nn.Linear(128, 128)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        # policy
        self.policy_1 = nn.Linear(512, 128)
        self.policy_2 = nn.Linear(128, 128)

        # Critic network
        self.value = nn.Linear(128, 1)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GNN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        if self.gnn == 'GCN':
            X_graph = self.GCN(X, A_in_Sparse)
            X_graph = F.relu(X_graph)
            X_graph = self.Graph_Dense(X_graph)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GAT':
            X_graph_1 = self.GAT_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.GAT_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GraphSAGE':
            X_graph_1 = self.SAGE_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.SAGE_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        F_concat = torch.cat((X_graph, X), 1)

        # policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value


# ------Graph Cost Critic Model------ #
class Graph_Cost_Critic_Model(nn.Module):
    def __init__(self, N, F, A, gnn):
        super(Graph_Cost_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.gnn = gnn

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # GNN
        if self.gnn == 'GCN':
            # self.GCN = GCNConv(128, 128)
            self.GCN = SGConv(256, 256, K=2, add_self_loops=False)
            self.Graph_Dense = nn.Linear(256, 256)
        elif self.gnn == 'GAT':
            heads = 4
            self.GAT_1 = GATConv(128, 128, heads=heads, add_self_loops=False)
            self.GAT_2 = GATConv(128 * heads, 128, heads=heads, add_self_loops=False)
            self.Graph_Dense = nn.Linear(128 * heads, 128)
        elif self.gnn == 'GraphSAGE':
            self.SAGE_1 = SAGEConv(128, 128)
            self.SAGE_2 = SAGEConv(128, 128)
            self.Graph_Dense = nn.Linear(128, 128)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        # policy
        self.policy_1 = nn.Linear(512, 128)
        self.policy_2 = nn.Linear(128, 128)

        # Critic network
        self.value = nn.Linear(128, 1)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GNN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        if self.gnn == 'GCN':
            X_graph = self.GCN(X, A_in_Sparse)
            X_graph = F.relu(X_graph)
            X_graph = self.Graph_Dense(X_graph)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GAT':
            X_graph_1 = self.GAT_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.GAT_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        elif self.gnn == 'GraphSAGE':
            X_graph_1 = self.SAGE_1(X, A_in_Sparse)
            X_graph_1 = F.relu(X_graph_1)
            X_graph_2 = self.SAGE_2(X_graph_1, A_in_Sparse)
            X_graph_2 = F.relu(X_graph_2)
            X_graph = self.Graph_Dense(X_graph_2)
            X_graph = F.relu(X_graph)
        else:
            raise ValueError(f"Unexpected value for GNN: {self.gnn}")

        F_concat = torch.cat((X_graph, X), 1)

        # policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value


# ------NonGraph Actor Model------ #
class NonGraph_Actor_Model(nn.Module):
    def __init__(self, N, F, A):
        super(NonGraph_Actor_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # policy
        self.policy_1 = nn.Linear(256, 128)
        self.policy_2 = nn.Linear(128, 128)

        # Actor network
        self.pi = nn.Linear(128, A)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, actions_mask=None):

        X_in, _, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # policy
        X_policy = self.policy_1(X)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        pi = self.pi(X_policy)
        pi = torch.mul(pi, mask)

        if actions_mask is not None:
            pi = pi + (actions_mask * 1e10 - 1e10)

        return pi


# ------NonGraph Critic Model------ #
class NonGraph_Critic_Model(nn.Module):
    def __init__(self, N, F, A):
        super(NonGraph_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # policy
        self.policy_1 = nn.Linear(256, 128)
        self.policy_2 = nn.Linear(128, 128)

        # Critic network
        self.value = nn.Linear(128, 1)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):

        X_in, _, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # policy
        X_policy = self.policy_1(X)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value


# ------NonGraph Cost Critic Model------ #
class NonGraph_Cost_Critic_Model(nn.Module):
    def __init__(self, N, F, A):
        super(NonGraph_Cost_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A

        # encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # policy
        self.policy_1 = nn.Linear(256, 128)
        self.policy_2 = nn.Linear(128, 128)

        # Critic network
        self.value = nn.Linear(128, 1)

        self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):

        X_in, _, RL_indice = datatype_transmission(observation, self.device)

        # encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # policy
        X_policy = self.policy_1(X)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        mask = torch.reshape(RL_indice, (self.num_agents, 1))

        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value
