import torch
from GRL_Library.agent import PPO_Lag_agent


def create_ppo_lag(params):
    N = params['n_vehicles']
    F = params['feature_dim']
    A = params['action_dim']
    Graph = params['Graph']
    assert isinstance(Graph, bool)
    if Graph:
        from GRL_Net.PPO_model import Graph_Actor_Model, Graph_Critic_Model, Graph_Cost_Critic_Model
        actor = Graph_Actor_Model(N=N, F=F, A=A, gnn=params['GNN'])
        critic = Graph_Critic_Model(N=N, F=F, A=A, gnn=params['GNN'])
        cost_critic = Graph_Cost_Critic_Model(N=N, F=F, A=A, gnn=params['GNN'])
    else:
        from GRL_Net.PPO_model import NonGraph_Actor_Model, NonGraph_Critic_Model, NonGraph_Cost_Critic_Model
        actor = NonGraph_Actor_Model(N=N, F=F, A=A)
        critic = NonGraph_Critic_Model(N=N, F=F, A=A)
        cost_critic = NonGraph_Cost_Critic_Model(N=N, F=F, A=A)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.00075, eps=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.00075, eps=0.001)
    cost_critic_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=0.00075, eps=0.001)

    # Discount factor
    gamma = 0.9
    # GAE factor
    gae_lambda = 0.95
    # Policy clip factor
    policy_clip = 0.2

    # Initialize GRL agent
    ppo_lag = PPO_Lag_agent.PPOLag(
        actor_model=actor,
        actor_optimizer=actor_optimizer,
        critic_model=critic,
        critic_optimizer=critic_optimizer,
        cost_critic_model=cost_critic,
        cost_critic_optimizer=cost_critic_optimizer,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=32,
        n_epochs=5,  # update times for one batch
        update_interval=128,  # update interval
        cost_threshold=1.0,
        eta=0.05,
        model_name="PPO_Lag_model",
        params=params
    )

    return actor, ppo_lag
