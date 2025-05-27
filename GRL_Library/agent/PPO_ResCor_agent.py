import torch
import numpy as np
import torch.nn.functional as F
import collections


class PPOResCorMemory(object):
    def __init__(self, batch_size):
        self.states = []
        self.actions_masks = []
        self.origin_probs = []
        self.dones = []
        self.risks = []
        self.alphas = []
        self.batch_size = batch_size

        self.origin_actions = []
        self.origin_action_probs = []
        self.origin_values = []
        self.origin_rewards = []

        self.safety_actions = []
        self.safety_action_probs = []
        self.safety_values = []
        self.safety_rewards = []

    def generate_batch(self, num_av):
        n_states = len(self.states)
        # risk-aware prioritized experience replay
        indices = np.arange(n_states, dtype=np.int64)
        weights_tensor = torch.stack(self.risks)[:, :num_av].sum(dim=1)
        weights = weights_tensor.cpu().numpy().astype(np.float32)
        probabilities = weights / weights.sum()
        sampled_indices = np.random.choice(indices, size=n_states, replace=True, p=probabilities)
        batches = [sampled_indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        return self.states, self.actions_masks, self.origin_probs, self.dones, self.risks, self.alphas, batches, \
            self.origin_actions, self.origin_action_probs, self.origin_values, self.origin_rewards, \
            self.safety_actions, self.safety_action_probs, self.safety_values, self.safety_rewards

    def store_memory(self, state, actions_mask, origin_prob, done, risk, alpha,
                     origin_action, origin_action_prob, origin_value, origin_reward,
                     safety_action, safety_action_prob, safety_value, safety_reward):

        origin_action = torch.as_tensor(origin_action, dtype=torch.long, device=torch.device("cpu"))
        safety_action = torch.as_tensor(safety_action, dtype=torch.long, device=torch.device("cpu"))
        origin_reward = torch.as_tensor(origin_reward, dtype=torch.float32, device=torch.device("cpu"))
        safety_reward = torch.as_tensor(safety_reward, dtype=torch.float32, device=torch.device("cpu"))
        risk = torch.as_tensor(risk, dtype=torch.float32, device=torch.device("cpu"))

        self.states.append(state)
        self.actions_masks.append(actions_mask)
        self.origin_probs.append(origin_prob)
        self.dones.append(done)
        self.risks.append(risk)
        self.alphas.append(alpha)

        self.origin_actions.append(origin_action)
        self.origin_action_probs.append(origin_action_prob)
        self.origin_values.append(origin_value)
        self.origin_rewards.append(origin_reward)

        self.safety_actions.append(safety_action)
        self.safety_action_probs.append(safety_action_prob)
        self.safety_values.append(safety_value)
        self.safety_rewards.append(safety_reward)

    def clear_memory(self):
        self.states = []
        self.actions_masks = []
        self.origin_probs = []
        self.dones = []
        self.risks = []
        self.alphas = []

        self.origin_actions = []
        self.origin_action_probs = []
        self.origin_values = []
        self.origin_rewards = []

        self.safety_actions = []
        self.safety_action_probs = []
        self.safety_values = []
        self.safety_rewards = []


class PPOResCor(object):
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
                 model_name,
                 safety_actor_model,
                 safety_actor_optimizer,
                 safety_critic_model,
                 safety_critic_optimizer,
                 safety_gamma=0.9,
                 safety_lambda=0.95,
                 safety_clip=0.2,
                 params=None):

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

        # ResCor model
        self.safety_actor_model = safety_actor_model
        self.safety_actor_optimizer = safety_actor_optimizer
        self.safety_critic_model = safety_critic_model
        self.safety_critic_optimizer = safety_critic_optimizer
        self.safety_gamma = safety_gamma
        self.safety_lambda = safety_lambda
        self.safety_clip = safety_clip
        # ResCor model transfer learning
        transfer = params.get('Transfer', False)
        self.freeze_safety = transfer

        self.params = params
        self.device = "cpu"

        # Replay buffer
        self.memory = PPOResCorMemory(self.batch_size)

        self.time_counter = 1

        self.loss_record = collections.deque(maxlen=100)

    def store_transition(self, state, actions_mask, origin_prob, done, risk, alpha,
                         origin_action, origin_action_prob, origin_value, origin_reward,
                         safety_action, safety_action_prob, safety_value, safety_reward):
        self.memory.store_memory(state, actions_mask, origin_prob, done, risk, alpha,
                                 origin_action, origin_action_prob, origin_value, origin_reward,
                                 safety_action, safety_action_prob, safety_value, safety_reward)

    def choose_action(self, observation, actions_mask, risk):
        # ------ main model ------ #
        logits = self.actor_model(observation, actions_mask=actions_mask)
        origin_probs = F.softmax(logits, dim=-1)
        origin_dist = torch.distributions.Categorical(origin_probs)

        origin_values = self.critic_model(observation)
        origin_values = torch.squeeze(origin_values)

        # ------ ResCor model ------ #
        safety_logits = self.safety_actor_model(observation, origin_probs.detach(), actions_mask=actions_mask)
        safety_probs = F.softmax(safety_logits, dim=-1)
        safety_dist = torch.distributions.Categorical(safety_probs)

        safety_values = self.safety_critic_model(observation, origin_probs.detach())
        safety_values = torch.squeeze(safety_values)

        # ------ residual connections ------ #
        weight_tensor = torch.tensor(risk, dtype=torch.float32)
        alpha = 0.2 + 0.6 * torch.sigmoid(100 * (weight_tensor - 0.7))
        final_probs = (1 - alpha.unsqueeze(-1)) * origin_probs + alpha.unsqueeze(-1) * safety_probs
        final_dist = torch.distributions.Categorical(final_probs)
        final_actions = final_dist.sample()

        origin_actions = final_actions.clone().detach()
        origin_action_probs = origin_dist.log_prob(origin_actions)

        safety_actions = final_actions.clone().detach()
        safety_action_probs = safety_dist.log_prob(safety_actions)

        return final_actions, origin_probs, alpha, \
            origin_actions, origin_action_probs, origin_values, \
            safety_actions, safety_action_probs, safety_values

    def test_action(self, observation, actions_mask, risk):
        # ------ main model ------ #
        logits = self.actor_model(observation, actions_mask=actions_mask)
        origin_probs = F.softmax(logits, dim=-1)

        # ------ ResCor model ------ #
        safety_logits = self.safety_actor_model(observation, origin_probs.detach(), actions_mask=actions_mask)
        safety_probs = F.softmax(safety_logits, dim=-1)

        # ------ residual connections ------ #
        weight_tensor = torch.tensor(risk, dtype=torch.float32)
        alpha = 0.2 + 0.6 * torch.sigmoid(100 * (weight_tensor - 0.7))
        final_probs = (1 - alpha.unsqueeze(-1)) * origin_probs + alpha.unsqueeze(-1) * safety_probs
        final_actions = torch.argmax(final_probs, dim=-1)

        return final_actions

    def learn(self):
        if self.time_counter % self.update_interval != 0:
            self.time_counter += 1
            return

        for _ in range(self.n_epochs):
            n_av = self.params['n_av']
            state_arr, actions_mask_arr, origin_prob_arr, done_arr, weight_arr, alpha_arr, batches, \
                origin_action_arr, origin_action_prob_arr, origin_value_arr, origin_reward_arr, \
                safety_action_arr, safety_action_prob_arr, safety_value_arr, safety_reward_arr = self.memory.generate_batch(n_av)

            values = origin_value_arr
            safety_values = safety_value_arr

            # ------ GAE of the main model ------ #
            advantage = torch.zeros(len(origin_reward_arr), len(origin_action_arr[0])).to(self.device)
            gae = torch.zeros(len(origin_action_arr[0])).to(self.device)
            for t in reversed(range(len(origin_reward_arr))):
                delta = origin_reward_arr[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) * (1 - int(done_arr[t])) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - int(done_arr[t])) * gae
                advantage[t] = gae

            if not self.freeze_safety:
                # ------ GAE of the ResCor model ------ #
                safety_advantage = torch.zeros(len(safety_reward_arr), len(safety_action_arr[0])).to(self.device)
                safety_gae = torch.zeros(len(safety_action_arr[0])).to(self.device)
                for t in reversed(range(len(safety_reward_arr))):
                    safety_delta = safety_reward_arr[t] + self.safety_gamma * (safety_values[t + 1] if t + 1 < len(safety_values) else 0) * (1 - int(done_arr[t])) - safety_values[t]
                    safety_gae = safety_delta + self.safety_gamma * self.safety_lambda * (1 - int(done_arr[t])) * safety_gae
                    safety_advantage[t] = safety_gae

            values = torch.stack(values)
            safety_values = torch.stack(safety_values)

            # Training for the collected samples
            for batch in batches:
                actor_loss_matrix = []
                critic_loss_matrix = []
                safety_actor_loss_matrix = []
                safety_critic_loss_matrix = []

                for i in batch:
                    veh_num = state_arr[i][2].sum()
                    mask = torch.tensor(state_arr[i][2], dtype=torch.float32, device=self.device)

                    # ------ main model actor ------ #
                    old_probs = origin_action_prob_arr[i].detach()
                    actions = origin_action_arr[i]

                    logits = self.actor_model(state_arr[i], actions_mask=actions_mask_arr[i])
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = torch.exp(new_probs - old_probs)
                    weighted_probs = advantage[i].detach() * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                         1 + self.policy_clip) * advantage[i].detach()
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                    actor_loss = torch.sum(actor_loss * weight_arr[i] * (1 - alpha_arr[i]) * mask) / veh_num
                    actor_loss_matrix.append(actor_loss)

                    # ------ main model critic ------ #
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = torch.squeeze(critic_value)

                    returns = advantage[i] + values[i]
                    returns = returns.detach()
                    critic_loss = F.smooth_l1_loss(returns, critic_value, reduction='none')
                    critic_loss = torch.sum(critic_loss * weight_arr[i] * (1 - alpha_arr[i]) * mask) / veh_num
                    critic_loss_matrix.append(critic_loss)

                    if not self.freeze_safety:
                        # ------ ResCor model actor ------ #
                        old_safety_probs = safety_action_prob_arr[i].detach()
                        safety_actions = safety_action_arr[i]

                        safety_logits = self.safety_actor_model(state_arr[i], origin_prob_arr[i].detach(), actions_mask=actions_mask_arr[i])
                        safety_probs = F.softmax(safety_logits, dim=-1)
                        safety_dist = torch.distributions.Categorical(safety_probs)

                        new_safety_probs = safety_dist.log_prob(safety_actions)
                        safety_prob_ratio = torch.exp(new_safety_probs - old_safety_probs)
                        weighted_safety_probs = safety_advantage[i].detach() * safety_prob_ratio
                        weighted_clipped_safety_probs = torch.clamp(safety_prob_ratio, 1 - self.safety_clip,
                                                                    1 + self.safety_clip) * safety_advantage[i].detach()
                        safety_actor_loss = -torch.min(weighted_safety_probs, weighted_clipped_safety_probs)
                        safety_actor_loss = torch.sum(safety_actor_loss * weight_arr[i] * alpha_arr[i] * mask) / veh_num
                        safety_actor_loss_matrix.append(safety_actor_loss)

                        # ------ ResCor model critic ------ #
                        safety_critic_value = self.safety_critic_model(state_arr[i], origin_prob_arr[i].detach())
                        safety_critic_value = torch.squeeze(safety_critic_value)

                        safety_returns = safety_advantage[i] + safety_values[i]
                        safety_returns = safety_returns.detach()
                        safety_critic_loss = F.smooth_l1_loss(safety_returns, safety_critic_value, reduction='none')
                        safety_critic_loss = torch.sum(safety_critic_loss * weight_arr[i] * alpha_arr[i] * mask) / veh_num
                        safety_critic_loss_matrix.append(safety_critic_loss)

                actor_loss_matrix = torch.stack(actor_loss_matrix)
                actor_loss_mean = torch.mean(actor_loss_matrix)

                critic_loss_matrix = torch.stack(critic_loss_matrix)
                critic_loss_mean = 0.5 * torch.mean(critic_loss_matrix)

                if not self.freeze_safety:
                    safety_actor_loss_matrix = torch.stack(safety_actor_loss_matrix)
                    safety_actor_loss_mean = torch.mean(safety_actor_loss_matrix)

                    safety_critic_loss_matrix = torch.stack(safety_critic_loss_matrix)
                    safety_critic_loss_mean = 0.5 * torch.mean(safety_critic_loss_matrix)

                self.actor_optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_optimizer.step()

                if not self.freeze_safety:
                    self.safety_actor_optimizer.zero_grad()
                    safety_actor_loss_mean.backward()
                    self.safety_actor_optimizer.step()

                    self.safety_critic_optimizer.zero_grad()
                    safety_critic_loss_mean.backward()
                    self.safety_critic_optimizer.step()

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
        save_path_safety_actor = save_path + "/" + self.model_name + "_safety_actor" + ".pt"
        save_path_safety_critic = save_path + "/" + self.model_name + "_safety_critic" + ".pt"
        torch.save(self.actor_model.state_dict(), save_path_actor)
        torch.save(self.critic_model.state_dict(), save_path_critic)
        torch.save(self.safety_actor_model.state_dict(), save_path_safety_actor)
        torch.save(self.safety_critic_model.state_dict(), save_path_safety_critic)

    def load_model(self, load_path):
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        load_path_safety_actor = load_path + "/" + self.model_name + "_safety_actor" + ".pt"
        load_path_safety_critic = load_path + "/" + self.model_name + "_safety_critic" + ".pt"
        self.actor_model.load_state_dict(torch.load(load_path_actor))
        self.critic_model.load_state_dict(torch.load(load_path_critic))
        self.safety_actor_model.load_state_dict(torch.load(load_path_safety_actor))
        self.safety_critic_model.load_state_dict(torch.load(load_path_safety_critic))
