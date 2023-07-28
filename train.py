import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# %%
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import model_car_env
#from model_car_env import ModelCar
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# %%
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

import time
env = None
while env is None:
    try:
        env = model_car_env.ModelCar()
    except:
        time.sleep(1)

# model = DQN("MlpPolicy", env, verbose=1,
#             learning_starts=10,
#             buffer_size = 4000,
#             learning_rate=0.01,
#             train_freq=1,
#             target_update_interval=50,
#             exploration_initial_eps=0.01,
#             exploration_final_eps=0.01,
#             device=1)
checkpoint_callback = CheckpointCallback(
  save_freq=100,
  save_path="/home/pistar/Desktop/JetRacer/DeepRacerModels/",
  name_prefix="deepracer_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model =PPO("MlpPolicy", env, n_steps=8, learning_rate=0.01, stats_window_size=10)

#model = PPO.load("/home/pistar/Desktop/JetRacer/deepracer_model_1200_steps", env=env)

model.learn(total_timesteps=10_000, callback=checkpoint_callback)
model.save("/home/pistar/Desktop/JetRacer/DeepRacerTrack")
# TODO save/load model
# TODO save/load replay buffer for offline RL

#vae = torch.load('/home/pistar/Desktop/JetRacer/dataset/dataset/vae.pth')
#model = PPO.load("/home/pistar/Desktop/JetRacer/DeepRacerModels/deepracer_model_1000_steps")

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(10_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

env.close()