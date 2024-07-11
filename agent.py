import torch
import numpy as np
from actor import ActorNetwork
from critic import CriticNetwork
from memory import ReplayBuffer


class TD3Agent(torch.nn.Module):
    def __init__(
        self,
        env_name,
        input_dims,
        n_actions,
        action_low_bounds,
        action_high_bounds,
        tau=5e-3,
        alpha=1e-4,
        beta=1e-3,
        gamma=0.99,
        batch_size=100,
        mem_size=int(1e6),
        warmup=int(1e3),
        update_interval=2,
        noise=0.1,
    ):
        super(TD3Agent, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_low_bounds = action_low_bounds
        self.action_high_bounds = action_high_bounds
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_interval = update_interval
        self.noise = noise

        self.learn_step_counter = 0
        self.time_step = 0

        self.memory = ReplayBuffer(input_dims, self.n_actions, buffer_length=mem_size)

        self.actor = ActorNetwork(
            input_dims,
            self.n_actions,
            lr=self.alpha,
            chkpt_path=f"weights/{env_name}_actor.pt",
        )
        self.target_actor = ActorNetwork(
            input_dims,
            self.n_actions,
            lr=self.alpha,
            chkpt_path=f"weights/{env_name}_target_actor.pt",
        )

        self.critic_1 = CriticNetwork(
            input_dims,
            self.n_actions,
            lr=self.beta,
            chkpt_path=f"weights/{env_name}_critic_1.pt",
        )
        self.critic_2 = CriticNetwork(
            input_dims,
            self.n_actions,
            lr=self.beta,
            chkpt_path=f"weights/{env_name}_critic_2.pt",
        )
        self.target_critic_1 = CriticNetwork(
            input_dims,
            self.n_actions,
            lr=self.beta,
            chkpt_path=f"weights/{env_name}_target_critic_1.pt",
        )
        self.target_critic_2 = CriticNetwork(
            input_dims,
            self.n_actions,
            lr=self.beta,
            chkpt_path=f"weights/{env_name}_target_critic_2.pt",
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, state):
        self.actor.eval()

        if self.time_step < self.warmup:
            mu = torch.randn(self.n_actions).to(self.actor.device) * self.noise
            # mu = torch.tensor(np.random.normal(scale=self.noise, size=self.n_actions))

        else:
            state = torch.Tensor(np.array(state)).to(self.actor.device)
            mu = self.actor(state).to(self.actor.device)

        # add gauss(0, 0.1) noise to deterministic output
        mu += torch.randn(mu.size()).to(self.actor.device) * self.noise

        # clamp noise to action space
        #
        # wait... the actor net will produce values -1, 1 due to tanh, so wouldn't 
        # I need to first multiply the action by the action max before adding noise?
        action_min = torch.tensor(self.action_low_bounds).to(self.actor.device)
        action_max = torch.tensor(self.action_high_bounds).to(self.actor.device)
        mu = torch.clamp(mu, action_min, action_max)

        self.time_step += 1
        self.actor.train()

        # return mu.cpu().detach().numpy()[0]
        return mu.cpu().detach().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.memory.sample(
            self.batch_size
        )
        states = torch.Tensor(states).to(self.actor.device)
        actions = torch.Tensor(actions).to(self.actor.device)
        next_states = torch.Tensor(next_states).to(self.actor.device)
        rewards = torch.Tensor(rewards).to(self.actor.device)
        done = torch.Tensor(done).to(self.actor.device).to(torch.bool)

        target_actions = self.target_actor(next_states)
        target_actions = target_actions + torch.clamp(
            torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        action_min = torch.tensor(self.action_low_bounds).to(self.actor.device)
        action_max = torch.tensor(self.action_high_bounds).to(self.actor.device)
        target_actions = torch.clamp(target_actions, action_min, action_max)

        target_c1_values = self.target_critic_1(next_states, target_actions)
        target_c2_values = self.target_critic_2(next_states, target_actions)

        target_c1_values[done] = 0.0
        target_c2_values[done] = 0.0
        target_c1_values = target_c1_values.view(-1)
        target_c2_values = target_c2_values.view(-1)
        target_values = torch.min(target_c1_values, target_c2_values)

        target = rewards + self.gamma * target_values
        target = target.view(self.batch_size, 1)  # add batch dim

        critic_1_values = self.critic_1(states, actions)
        critic_2_values = self.critic_2(states, actions)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        critic_1_loss = torch.nn.functional.mse_loss(target, critic_1_values)
        critic_2_loss = torch.nn.functional.mse_loss(target, critic_2_values)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states))
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in actor_params:
            actor_params[name] = (
                tau * actor_params[name].clone()
                + (1 - tau) * target_actor_params[name].clone()
            )
        self.target_actor.load_state_dict(actor_params)

        critic_params = dict(self.critic_1.named_parameters())
        target_critic_params = dict(self.target_critic_1.named_parameters())
        for name in critic_params:
            critic_params[name] = (
                tau * critic_params[name].clone()
                + (1 - tau) * target_critic_params[name].clone()
            )
        self.target_critic_1.load_state_dict(critic_params)

        critic_params = dict(self.critic_2.named_parameters())
        target_critic_params = dict(self.target_critic_2.named_parameters())
        for name in critic_params:
            critic_params[name] = (
                tau * critic_params[name].clone()
                + (1 - tau) * target_critic_params[name].clone()
            )
        self.target_critic_2.load_state_dict(critic_params)
