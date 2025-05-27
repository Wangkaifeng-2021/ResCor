import torch
import numpy as np
import torch.nn.functional as F
import collections


class PPOMemory(object):
    def __init__(self, batch_size):
        self.states = []
        self.actions_masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions_masks, self.actions, self.probs, self.vals, self.rewards, self.dones, batches

    def store_memory(self, state, actions_mask, action, probs, vals, reward, done):
        action = torch.as_tensor(action, dtype=torch.long, device=torch.device("cpu"))
        reward = torch.as_tensor(reward, dtype=torch.float32, device=torch.device("cpu"))

        self.states.append(state)
        self.actions_masks.append(actions_mask)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO(object):
    def __init__(self,
                 actor_model,
                 actor_optimizer,
                 critic_model,
                 critic_optimizer,
                 gamma,
                 gae_lambda,
                 policy_clip,
                 batch_size,
                 n_epochs,
                 update_interval,
                 model_name):

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.model_name = model_name

        self.device = "cpu"

        # Replay buffer
        self.memory = PPOMemory(self.batch_size)

        self.time_counter = 1

        self.loss_record = collections.deque(maxlen=100)

    def store_transition(self, state, actions_mask, action, probs, vals, reward, done):
        self.memory.store_memory(state, actions_mask, action, probs, vals, reward, done)

    def choose_action(self, observation, actions_mask=None):
        logits = self.actor_model(observation, actions_mask=actions_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        probs = dist.log_prob(action)
        value = self.critic_model(observation)
        value = torch.squeeze(value)

        return action, probs, value

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
            state_arr, actions_mask_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batch()

            values = vals_arr

            # ------ GAE ------ #
            advantage = torch.zeros(len(reward_arr), len(action_arr[0])).to(self.device)
            gae = torch.zeros(len(action_arr[0])).to(self.device)
            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) * (1 - int(dones_arr[t])) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae
                advantage[t] = gae

            values = torch.stack(values)

            # Training for the collected samples
            for batch in batches:
                actor_loss_matrix = []
                critic_loss_matrix = []

                for i in batch:
                    veh_num = state_arr[i][2].sum()
                    mask = torch.tensor(state_arr[i][2], dtype=torch.float32, device=self.device)

                    # ------ actor ------ #
                    old_probs = old_prob_arr[i].detach()
                    actions = action_arr[i]

                    logits = self.actor_model(state_arr[i], actions_mask=actions_mask_arr[i])
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = torch.exp(new_probs - old_probs)
                    weighted_probs = advantage[i].detach() * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                         1 + self.policy_clip) * advantage[i].detach()
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

                actor_loss_matrix = torch.stack(actor_loss_matrix)
                actor_loss_mean = torch.mean(actor_loss_matrix)

                critic_loss_matrix = torch.stack(critic_loss_matrix)
                critic_loss_mean = 0.5 * torch.mean(critic_loss_matrix)

                self.actor_optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_optimizer.step()

                # Save loss
                self.loss_record.append(float((actor_loss_mean + critic_loss_mean).detach().cpu().numpy()))

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
