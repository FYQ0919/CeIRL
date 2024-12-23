import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional
import torch.nn.functional as F

# from DreamWaQ.modules import ActorCritic
from FTVQ.modules import ActorCritic
from FTVQ.storage import RolloutStorage
from FTVQ.training_config import RunnerCfg


class PPO:
    def __init__(
        self,
        actor_critic: ActorCritic,
        cfg: RunnerCfg.algorithm,
        device="cpu",
    ):
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.cfg = cfg
        self.device = device
        
        self.learning_rate = self.cfg.learning_rate
        self.entropy_coef = self.cfg.entropy_coef
        self.kl_weight = self.cfg.kl_weight

        # for A2C
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.cfg.learning_rate
        )
        # for VAE
        self.vae_optimizer = optim.Adam(
            self.actor_critic.vae.parameters(),
            lr=self.cfg.vae_learning_rate,
        )

        self.transition = RolloutStorage.Transition()
        self.use_forward = True
        self.use_contrastive = True

        if self.use_forward:
            self.self_supervised_optimizer = torch.optim.Adam([
                {'params': self.actor_critic.forward_model.parameters(), 'lr': self.cfg.adaptation_module_learning_rate,
                 'lr_name': 'adaptation_module_learning_rate', },
            ])

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            privileged_obs_shape,
            obs_history_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, priviledged_obs, obs_history, base_vel, cv):
        # Compute the actions and values
        self.transition.actions, _, _ = self.actor_critic.act_student(
            obs, obs_history, cv
        )
        self.transition.actions = self.transition.actions.detach()
        self.transition.values = self.actor_critic.evaluate(
            obs, priviledged_obs, base_vel
        ).detach()
        self.transition.actions_log_prob = (
            self.actor_critic.get_actions_log_prob(
                self.transition.actions
            ).detach()
        )
        self.transition.action_mean = (
            self.actor_critic.action_mean.detach()
        )
        self.transition.action_sigma = (
            self.actor_critic.action_std.detach()
        )
        # need to record obs, privileged_obs, base_vel before env.step()
        self.transition.observations = obs
        # self.transition.critic_observations = obs
        self.transition.privileged_observations = priviledged_obs
        self.transition.obs_histories = obs_history
        self.transition.base_vel = base_vel
        self.transition.cv = cv
        return self.transition.actions

    def process_env_step(self, rewards, dones, next_obs, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        self.transition.next_observations = next_obs
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.cfg.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(
        self,
        last_critic_obs,
        last_critic_privileged_obs,
        last_base_vel,
    ):
        last_values = self.actor_critic.evaluate(
            last_critic_obs,
            last_critic_privileged_obs,
            last_base_vel,
        ).detach()
        self.storage.compute_returns(
            last_values, self.cfg.gamma, self.cfg.lam
        )

    def update(self):
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0
        mean_forward_loss = 0
        mean_contrastive_loss = 0

        generator = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches,
            self.cfg.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            privileged_obs_batch,
            obs_history_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            env_bins_batch,
            base_vel_batch,
            dones_batch,
            next_obs_batch, cv
        ) in generator:
            _, forward_predict,latent = self.actor_critic.act_student(
                obs_batch, obs_history_batch, cv
            )
            actions_log_prob_batch = (
                self.actor_critic.get_actions_log_prob(
                    actions_batch
                )
            )
            value_batch = self.actor_critic.evaluate(
                obs_batch,
                privileged_obs_batch,
                base_vel_batch,
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.cfg.desired_kl != None and self.cfg.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        print("Learning_rate:",self.learning_rate,"Changed cause by KL:",kl_mean)
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(2e-3, self.learning_rate * 1.2)
                        print("Learning_rate:",self.learning_rate,"Changed cause by KL:",kl_mean)
                        
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            
            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg.clip_param,
                                                                            1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.cfg.clip_param,
                                                                                                self.cfg.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            if self.use_contrastive:
                # Compute contrastive loss
                leg_mask = privileged_obs_batch[:,-12:]
                # print(leg_mask[0])

                # 找到 leg_mask 中值为 0 的位置
                zero_positions = (leg_mask == 0).nonzero()

                label = torch.ones((leg_mask.shape[0], 1), dtype=torch.long, device=self.device) * 12

                # 如果找到了零值位置，则更新输出张量中对应位置的值为找到的第一个零值的索引
                if zero_positions.numel() > 0:
                    label[zero_positions[:, 0], 0] = zero_positions[:, 1]

                contrastive_loss = self.compute_contrastive_loss(latent, label)

                loss += self.cfg.contrastive_loss_coef * contrastive_loss
                mean_contrastive_loss += contrastive_loss.item()


            if self.use_forward:
                for _ in range(self.cfg.num_adaptation_module_substeps):
                    latent = self.actor_critic.get_latent(obs_history_batch, cv)
                    predicted_next_obs = self.actor_critic.forward_model.forward(latent)
                    mask = torch.logical_not(dones_batch).squeeze()
                    forward_loss = (F.mse_loss(predicted_next_obs[mask], next_obs_batch[mask]))
                    # leg_mask_tensor = leg_mask[mask].requires_grad_(True)
                    # privileged_leg_mask_tensor = privileged_obs_batch[mask][:, -12:].requires_grad_(True)

                    # mask_loss = F.mse_loss(leg_mask_tensor, privileged_leg_mask_tensor)
                    loss = loss + forward_loss
                    mean_forward_loss += forward_loss.item()
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()


                    # leg_mask_tensor = leg_mask[mask].requires_grad_(True)
                    # privileged_obs_batch_tensor = privileged_obs_batch[mask][:, -4:].requires_grad_(True)
                    # print(leg_mask_tensor, privileged_obs_batch[mask])
                    # mask_loss = F.mse_loss(leg_mask_tensor, privileged_obs_batch_tensor)

                    # loss = forward_loss
                    # # loss = forward_loss
                    # self.self_supervised_optimizer.zero_grad()
                    # loss.backward()
                    # self.self_supervised_optimizer.step()
                    # mean_forward_loss += forward_loss.item()


            for _ in range(self.cfg.num_adaptation_module_substeps):
                self.vae_optimizer.zero_grad()
                latent_batch = self.actor_critic.adaptation_module(obs_history_batch)
                vae_loss_dict = self.actor_critic.vae.loss_fn(
                    latent_batch,
                    obs_history_batch,
                    next_obs_batch,
                    base_vel_batch,
                    cv,
                    self.kl_weight,
                )
                valid = (dones_batch == 0).squeeze()
                vae_loss = torch.mean(
                    vae_loss_dict["loss"][valid]
                )

                vae_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor_critic.vae.parameters(),
                    self.cfg.max_grad_norm,
                )
                self.vae_optimizer.step()
                with torch.no_grad():
                    recons_loss = torch.mean(
                        vae_loss_dict["recons_loss"][valid]
                    )
                    vel_loss = torch.mean(
                        vae_loss_dict["vel_loss"][valid]
                    )
                    kld_loss = torch.mean(
                        vae_loss_dict["kld_loss"][valid]
                    )

                mean_recons_loss += recons_loss.item()
                mean_vel_loss += vel_loss.item()
                mean_kld_loss += kld_loss.item()

        num_updates = (
            self.cfg.num_learning_epochs
            * self.cfg.num_mini_batches
        )
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_forward_loss /= (num_updates * self.cfg.num_adaptation_module_substeps)
        mean_entropy_loss /= num_updates
        mean_recons_loss /= (
            num_updates * self.cfg.num_adaptation_module_substeps
        )
        mean_vel_loss /= (
            num_updates * self.cfg.num_adaptation_module_substeps
        )
        mean_kld_loss /= (
            num_updates * self.cfg.num_adaptation_module_substeps
        )
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy_loss,
            mean_recons_loss,
            mean_vel_loss,
            mean_kld_loss,
            mean_forward_loss,
            mean_contrastive_loss
        )

    def compute_contrastive_loss(self, latent, labels, temperature=1.0):
        # Get batch size
        batch_size = latent.shape[0]

        latent = latent.reshape(batch_size, latent.shape[1] * latent.shape[2])

        # Compute similarity scores
        similarity_scores = torch.matmul(latent, latent.T) / temperature

        # Generate labels to support positive samples with the same label
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Labels length mismatch")

        mask = torch.eq(labels, labels.T).float().to(latent.device)

        # Compute log probabilities
        log_prob = similarity_scores - torch.logsumexp(similarity_scores, dim=1, keepdim=True)

        # Compute InfoNCE loss
        # Consider only positive samples in the mask
        loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # Replace NaN values with zero
        loss[torch.isnan(loss)] = 0

        # Compute average loss
        loss = loss.mean()

        return loss

