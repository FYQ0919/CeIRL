import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from DreamWaQ.utils import (
    get_activation,
    MultivariateGaussianDiagonalCovariance,
    init_orhtogonal,
)
from .state_estimator import VAE
from .forward_model import MLPForward, MLPFFT

from graphviz import Digraph
from torch.autograd import Variable, Function

class ActorCritic(nn.Module):
    def __init__(
        self,
        num_obs,
        num_privileged_obs,
        num_actions=12,
        num_latent=16,
        num_history=20,
        activation="elu",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 64],
        decoder_hidden_dims=[64, 128, 256],
        init_noise_std=1.0,
        error_dim=8,
        fft_len=20,
        use_forward=True,
        use_fft=True,
    ):
        super().__init__()

        self.activation = get_activation(activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.use_forward = use_forward
        self.use_fft = use_fft

        self.vae = VAE(
            num_obs=num_obs,
            num_history=num_history,
            num_latent=num_latent,
            activation=self.activation,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        self.predict_obs = []
        self.obs = []
        # print(self.vae)
        estimated_vel = 3
        fft_input_size = fft_len * num_obs * 2
        history_size = num_obs * num_history
        if self.use_fft:
            input_size = num_obs + num_latent + error_dim + num_latent // 4 + estimated_vel + 12
        else:
            input_size = num_obs + num_latent + error_dim + estimated_vel + 12

        if self.use_forward:
            self.forward_model = MLPForward(num_obs, num_latent, num_actions, activation)
            self.error_encoder = nn.Sequential(
                nn.Linear(num_obs, 64),
                self.activation,
                nn.Linear(64, error_dim),
            )
        if self.use_forward:
            self.fft_model = MLPFFT(fft_input_size, num_latent // 4, num_actions, activation)

        self.actor = Actor(
            input_size=input_size,
            output_size=num_actions,
            activation=self.activation,
            hidden_dims=actor_hidden_dims,
        )

        self.adaptation_module = nn.Sequential(
            nn.Linear(num_obs, num_latent * 4),
            self.activation,
            nn.Linear(num_latent * 4, num_latent * 2),
            self.activation,
            nn.Linear(num_latent * 2, num_latent),
        )

        """
        :input = obs + latent + estimated_vel
        :output = actions
        """
        print(self.actor)
        real_vel = 3
        self.critic = Critic(
            input_size=num_obs + real_vel + num_privileged_obs,
            output_size=1,
            activation=self.activation,
            hidden_dims=critic_hidden_dims,
        )
        """
        :input = obs + real_vel + privileged_obs
        :output = reward
        """
        print(self.critic)
        self.distribution = None
        
        # self.distribution = MultivariateGaussianDiagonalCovariance(
        #     dim=num_actions,
        #     init_std=init_noise_std,
        # )
        # """distribution of noise for action"""
        # 在确定代码无误后，禁用参数验证可以略微提高性能
        Normal.set_default_validate_args = False

        self.vae.apply(init_orhtogonal)
        # self.actor.apply(init_orhtogonal)
        # self.critic.apply(init_orhtogonal)

    #! rollout 的时候需要随机性, 这里是 bootstrap
    def act_student(self, obs, obs_history, cv):
        """
        :obs_dict: obs, obs_history
        :return distribution.sample()
        :其中std会在模型的训练过程中自动调整
        """
        latent = self.adaptation_module(obs_history)
        latent_and_history = torch.cat([latent, obs_history], dim=-1)
        [z, vel],[latent_mu, latent_var, vel_mu, vel_var] = self.vae.sample(latent_and_history, cv)
        latent_and_estimatedVEL = torch.cat([z, vel], dim=1)
        if self.use_forward:
            forward_predict, leg_mask = self.forward_model(z)
            predict_error = forward_predict - obs
            error_latent = self.error_encoder(predict_error)

            if self.use_fft:
                fft_result = torch.fft.fft(obs_history)
                amplitude = torch.abs(fft_result)
                phase = torch.angle(fft_result)
                amplitude = amplitude.reshape(amplitude.shape[0], -1)
                phase = phase.reshape(phase.shape[0], -1)
                fft_latent = self.fft_model(torch.cat([amplitude, phase], dim=-1))
                input = torch.cat([obs, latent_and_estimatedVEL, error_latent, fft_latent, leg_mask], dim=1)
            else:
                input = torch.cat([obs, latent_and_estimatedVEL, error_latent, leg_mask], dim=1)

        else:
            input = torch.cat([obs, latent_and_estimatedVEL], dim=1)
        if torch.isnan(latent_and_estimatedVEL).any():
            raise ValueError("CENet ocurrs nan")
        self.pre_out = latent_and_estimatedVEL.detach()
        self.pre_latent_mu = latent_mu.detach()
        self.pre_latent_var = latent_var.detach()
        self.pre_vel_mu = vel_mu.detach()
        self.pre_vel_var = vel_var.detach()
        self.pre_std = self.std.detach()
        action_mean = self.actor.forward(input)
        if torch.isnan(action_mean).any():

            raise ValueError("action_mean ocurrs nan")
        self.update_distribution(action_mean)
        if self.use_forward:
            return self.distribution.sample(), forward_predict, latent
        else:
            return self.distribution.sample()

    # def act_expert(self, obs, privileged_obs, obs_history, vel):
    #     """obs_dict: obs, obs_history, privileged_obs"""
    #     latent_mu, _ = self.vae.inference(obs_history)
    #     latent = torch.cat([latent_mu, vel], dim=1)
    #     action_mean = self.actor.forward(obs, latent)
    #     self.distribution.update(action_mean)
    #     return self.distribution.sample()

    def reset(self, dones=None):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        latent = self.adaptation_module(obs_dict["obs_history"])
        np.save("fl_latent2",latent.detach().cpu().numpy())
        latent_and_history = torch.cat([latent, obs_dict["obs_history"]],dim=-1)
        latent_mu, vel_mu = self.vae.inference(latent_and_history)
        if self.use_forward:
            forward_predict, leg_mask = self.forward_model(latent_mu)
            predict_error = forward_predict - obs_dict["obs"]
            # self.obs.append(obs_dict["obs"].detach().cpu().numpy()[0,9:33])
            # self.predict_obs.append(forward_predict.detach().cpu().numpy()[0,9:33])
            # np.save("predict_obs1",np.array(self.predict_obs))
            # np.save("obs1",np.array(self.obs))
            error_latent = self.error_encoder(predict_error)
            latent = torch.cat([latent_mu, vel_mu], dim=1)
            if self.use_fft:
                fft_result = torch.fft.fft(obs_dict["obs_history"])
                amplitude = torch.abs(fft_result)
                phase = torch.angle(fft_result)
                amplitude = amplitude.reshape(amplitude.shape[0], -1)
                phase = phase.reshape(phase.shape[0], -1)
                fft_latent = self.fft_model(torch.cat([amplitude, phase], dim=-1))
                input = torch.cat([obs_dict["obs"], latent, error_latent, fft_latent, leg_mask], dim=1)
            else:
                input = torch.cat([obs_dict["obs"], latent, error_latent, leg_mask], dim=1)
        else:
            latent = torch.cat([latent_mu, vel_mu], dim=1)
            input = torch.cat([obs_dict["obs"], latent],dim=1)
        return self.actor.forward(input)

    def evaluate(self, obs, privileged_observations, vel):
        obs = torch.cat([obs, vel], dim=-1)
        input = torch.cat([obs, privileged_observations],dim=1)
        value = self.critic.forward(input)
        return value
    
    def update_distribution(self, mean):
        self.distribution = Normal(mean, mean*0. + self.std)

    def get_latent(self, obs_history, cv):
        latent = self.adaptation_module(obs_history)
        latent_and_history = torch.cat([latent, obs_history], dim=-1)
        [z, vel], [latent_mu, latent_var, vel_mu, vel_var] = self.vae.sample(latent_and_history, cv)

        return z

class Actor(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


class Critic(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


