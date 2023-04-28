from pytorch_lightning import LightningModule, Trainer
import copy
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from base64 import b64encode
from torch import Tensor, nn
from torch.nn import functional as F
import drl_project.brain_pyt as brain_pyt
import environment
class DQN(LightningModule):

    #initialize
    def __init__(self, policy=brain_pyt.epsilon_greedy, capacity=100000, batch_size=32, lr=1e-3, hidden_state=[64, 32],
                 gamma=0.99, loss_fn=F.smooth_l1_loss, eps_start=1.0, eps_end=0.01, eps_last_episode=100, sync_rate=10, samples_per_epoch=10000):
        super().__init__()
        obs_size = 3
        n_actions = 5

        self.qnet = brain_pyt.Brain(hidden_size=hidden_state, input_size=obs_size, num_actions=n_actions)
        self.target_q_net = copy.deepcopy(self.qnet)
        self.policy = policy
        self.replay_buffer = brain_pyt.ReplayBuffer(capacity)

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f"{len(self.buffer)} samples in experience, Filling.....")
            self.play_episode(epsilon=self.hparams.eps_start)
        def play_episode(self, epsilon=0.0):
            env = environment.Environment()
            state = env.reset()
            done = False
            while not done:
                action = self.policy(self.qnet, state, epsilon)
                next_state, reward, done, _ = env.update_env(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
        #Forward

