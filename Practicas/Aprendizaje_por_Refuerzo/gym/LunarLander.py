import gymnasium as gym
env = gym.make("CarRacing-v2", domain_randomize=True)

# normal reset, this changes the colour scheme by default
env.reset()

# reset with colour scheme change
env.reset(options={"randomize": True})

# reset with no colour scheme change
env.reset(options={"randomize": False})