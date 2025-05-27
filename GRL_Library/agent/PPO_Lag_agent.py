import torch
import numpy as np
import torch.nn.functional as F
import collections


class PPOLagMemory(object):
    def __init__(self, batch_size):
        self.states = []
        self.actions_masks = []
        self.probs = []
        self.vals = []
        self.cost_vals = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions_masks, self.actions, self.probs, self.vals, self.cost_vals, self.rewards, self.costs, self.dones, batches

    def store_memory(self, state, actions_mask, action, probs, vals, cost_vals, reward, cost, done):
        action = torch.as_tensor(action, dtype=torch.long, device=torch.device("cpu"))
        reward = torch.as_tensor(reward, dtype=torch.float32, device=torch.device("cpu"))
        cost = torch.as_tensor(cost, dtype=torch.float32, device=torch.device("cpu"))

        self.states.append(state)
        self.actions_masks.append(actions_mask)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.cost_vals.append(cost_vals)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.vals = []
        self.cost_vals = []


class PPOLag(object):
    def __init__(self,
                 actor_model,
                 actor_optimizer,
                 critic_model,
                 critic_optimizer,
                 cost_critic_model,
                 cost_critic_optimizer,
                 gamma,
                 gae_lambda,
                 policy_clip,
                 batch_size,
                 n_epochs,
                 update_interval,
                 cost_threshold,
                 eta,
                 model_name,
                 params=None):

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.cost_critic_model = cost_critic_model
        self.cost_critic_optimizer = cost_critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.lambda_ = torch.tensor(1.0)
        self.cost_threshold = cost_threshold
        self.eta = eta
        self.model_name = model_name
        self.params = params

        self.device = "cpu"

        # Replay buffer
        self.memory = PPOLagMemory(self.batch_size)

        self.time_counter = 1

        self.loss_record = collections.deque(maxlen=100)

    def store_transition(self, state, actions_mask, action, probs, vals, cost_vals, reward, cost, done):
        self.memory.store_memory(state, actions_mask, action, probs, vals, cost_vals, reward, cost, done)

    def choose_action(self, observation, actions_mask=None):
        logits = self.actor_model(observation, actions_mask=actions_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        probs = dist.log_prob(action)
        value = self.critic_model(observation)
        value = torch.squeeze(value)
        cost_value = self.cost_critic_model(observation)
        cost_value = torch.squeeze(cost_value)

        return action, probs, value, cost_value

    def test_action(self, observation, actions_mask=None):
        logits = self.actor_model(observation, actions_mask=actions_mask)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)

        return action

    def learn(self):
        if self.time_counter % self.update_interval != 0:
            self.time_counter += 1
            return

        for _ in range(self.n_epochs):
            state_arr, actions_mask_arr, action_arr, old_prob_arr, vals_arr, cost_vals_arr, reward_arr, costs_arr, dones_arr, batches = self.memory.generate_batch()

            n_av = self.params['n_av']
            costs_tensor = torch.stack(costs_arr)
            costs_tensor_sums = costs_tensor[:, :n_av].sum(dim=1)
            avg_cost = torch.mean(costs_tensor_sums)
            self.lambda_ = torch.clamp(
                self.lambda_ + self.eta * (avg_cost - self.cost_threshold),
                min=0.0
            )

            values = vals_arr
            cost_values = cost_vals_arr

            # ------ GAE ------ #
            advantage = torch.zeros(len(reward_arr), len(action_arr[0])).to(self.device)
            gae = torch.zeros(len(action_arr[0])).to(self.device)
            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) * (1 - int(dones_arr[t])) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae
                advantage[t] = gae

            # ------ GAE (cost) ------ #
            advantage_cost = torch.zeros(len(costs_arr), len(action_arr[0])).to(self.device)
            gae_cost = torch.zeros(len(action_arr[0])).to(self.device)
            for t in reversed(range(len(costs_arr))):
                delta_cost = costs_arr[t] + self.gamma * (cost_values[t + 1] if t + 1 < len(cost_values) else 0) * (1 - int(dones_arr[t])) - cost_values[t]
                gae_cost = delta_cost + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae_cost
                advantage_cost[t] = gae_cost

            adjusted_advantage = advantage - self.lambda_ * advantage_cost

            values = torch.stack(values)
            cost_values = torch.stack(cost_values)

            # Training for the collected samples
            for batch in batches:
                actor_loss_matrix = []
                critic_loss_matrix = []
                cost_critic_loss_matrix = []

                for i in batch:
                    veh_num = state_arr[i][2].sum()
                    mask = torch.tensor(state_arr[i][2], dtype=torch.float32, device=self.device)

                    current_advantage = adjusted_advantage[i].detach()

                    # ------ actor ------ #
                    old_probs = old_prob_arr[i].detach()
                    actions = action_arr[i]

                    logits = self.actor_model(state_arr[i], actions_mask=actions_mask_arr[i])
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = torch.exp(new_probs - old_probs)
                    weighted_probs = current_advantage * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                         1 + self.policy_clip) * current_advantage
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                    actor_loss = torch.sum(actor_loss * mask) / veh_num
                    actor_loss_matrix.append(actor_loss)

                    # ------ critic ------ #
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = torch.squeeze(critic_value)

                    returns = advantage[i] + values[i]
                    returns = returns.detach()
                    critic_loss = F.smooth_l1_loss(returns, critic_value, reduction='none')
                    critic_loss = torch.sum(critic_loss * mask) / veh_num
                    critic_loss_matrix.append(critic_loss)

                    # ------ cost critic ------ #
                    cost_critic_value = self.cost_critic_model(state_arr[i])
                    cost_critic_value = torch.squeeze(cost_critic_value)

                    cost_returns = advantage_cost[i] + cost_values[i]
                    cost_returns = cost_returns.detach()
                    cost_critic_loss = F.smooth_l1_loss(cost_returns, cost_critic_value, reduction='none')
                    cost_critic_loss = torch.sum(cost_critic_loss * mask) / veh_num
                    cost_critic_loss_matrix.append(cost_critic_loss)

                actor_loss_matrix = torch.stack(actor_loss_matrix)
                actor_loss_mean = torch.mean(actor_loss_matrix)

                critic_loss_matrix = torch.stack(critic_loss_matrix)
                critic_loss_mean = 0.5 * torch.mean(critic_loss_matrix)

                cost_critic_loss_matrix = torch.stack(cost_critic_loss_matrix)
                cost_critic_loss_mean = 0.5 * torch.mean(cost_critic_loss_matrix)

                self.actor_optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_optimizer.step()

                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss_mean.backward()
                self.cost_critic_optimizer.step()

                # Save loss
                self.loss_record.append(float((actor_loss_mean + critic_loss_mean + cost_critic_loss_mean).detach().cpu().numpy()))

        self.memory.clear_memory()
        self.time_counter += 1

    def get_statistics(self):
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        return [loss_statistics]

    def save_model(self, save_path):
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic = save_path + "/" + self.model_name + "_critic" + ".pt"
        torch.save(self.actor_model.state_dict(), save_path_actor)
        torch.save(self.critic_model.state_dict(), save_path_critic)

    def load_model(self, load_path):
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        self.actor_model.load_state_dict(torch.load(load_path_actor))
        self.critic_model.load_state_dict(torch.load(load_path_critic))
