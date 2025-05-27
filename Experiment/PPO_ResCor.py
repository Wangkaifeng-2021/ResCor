import torch
from GRL_Library.agent import PPO_ResCor_agent


def create_ppo_rescor(params):
    N = params['n_vehicles']
    F = params['feature_dim']
    A = params['action_dim']
    Graph = params['Graph']
    assert isinstance(Graph, bool)
    if Graph:
        from GRL_Net.PPO_model import Graph_Actor_Model, Graph_Critic_Model
        actor = Graph_Actor_Model(N=N, F=F, A=A, gnn=params['GNN'])
        critic = Graph_Critic_Model(N=N, F=F, A=A, gnn=params['GNN'])
    else:
        from GRL_Net.PPO_model import NonGraph_Actor_Model, NonGraph_Critic_Model
        actor = NonGraph_Actor_Model(N=N, F=F, A=A)
        critic = NonGraph_Critic_Model(N=N, F=F, A=A)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.00075, eps=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.00075, eps=0.001)

    # Discount factor
    gamma = 0.9
    # GAE factor
    gae_lambda = 0.95
    # Policy clip factor
    policy_clip = 0.2

    from GRL_Net.SafetyNet import SafetyActorModel, SafetyCriticModel
    safety_actor = SafetyActorModel(N=N, F=F, A=A)
    safety_critic = SafetyCriticModel(N=N, F=F, A=A)

    safety_actor_optimizer = torch.optim.Adam(safety_actor.parameters(), lr=0.00075, eps=0.001)
    safety_critic_optimizer = torch.optim.Adam(safety_critic.parameters(), lr=0.00075, eps=0.001)

    # Initialize GRL agent
    ppo_rescor = PPO_ResCor_agent.PPOResCor(
        actor_model=actor,
        actor_optimizer=actor_optimizer,
        critic_model=critic,
        critic_optimizer=critic_optimizer,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=32,
        n_epochs=3,  # update times for one batch
        update_interval=256,  # update interval
        model_name="PPO_ResCor_model",
        safety_actor_model=safety_actor,
        safety_actor_optimizer=safety_actor_optimizer,
        safety_critic_model=safety_critic,
        safety_critic_optimizer=safety_critic_optimizer,
        safety_gamma=0.9,
        safety_lambda=0.95,
        safety_clip=0.2,
        params=params
    )

    return actor, ppo_rescor
