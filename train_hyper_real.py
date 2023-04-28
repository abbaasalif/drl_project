# %%
import os
import numpy as np
import random as rn
import real_env
import brain
import dqn
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune import CLIReporter
# from tqdm import tqdm
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting the parameters
# epsilon is exploration parameter
epsilon = 0.3
number_actions=5
direction_boundary = (number_actions -1)/2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5


# choosing the model
train = True

# %%
def train_model(config):
    # building the environment by simply creating an object of the environment class
    env = real_env.Environment(optimal_temperature = (18.0,24.0), initial_month = 0, initial_bytes_received=0, initial_cpu_percent=0)
    # building the brain by simpy creating an object of the brain class
    brain_model = brain.Brain(learning_rate = config['learning_rate'], weight_decay=config['weight_decay'] , layers=config['layers'],num_actions=5, dropout=config['dropout'], activation=config['activation'], optimizer = config['optimizer'])
    # building the DQN model by simpy creating an object of the DQN class
    dqn_model = dqn.DQN(max_memory = max_memory, discount = config['discount_factor'])
    env.train = train
    model = brain_model.model
    early_stopping = True
    patience = 10
    best_total_reward = -np.inf
    patience_count = 0
    if (env.train):
        # STARTING THE LOOP OVER ALL THE EPOCHS (1 Epoch = 5 Months)
        for epoch in range(1, number_epochs):
            # INITIALIAZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
            total_reward = 0
            loss = 0.
            new_month = np.random.randint(0, 12)
            env.reset(new_month = new_month)
            game_over = False
            current_state, _, _ = env.observe()
            timestep = 0
            # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
            while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
                # PLAYING THE NEXT ACTION BY EXPLORATION
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, number_actions)
                    if (action - direction_boundary < 0):
                        direction = -1
                    else:
                        direction = 1
                    energy_ai = abs(action - direction_boundary) * temperature_step
                # PLAYING THE NEXT ACTION BY INFERENCE
                else:
                    q_values = model.predict(current_state)
                    action = np.argmax(q_values[0])
                    if (action - direction_boundary < 0):
                        direction = -1
                    else:
                        direction = 1
                    energy_ai = abs(action - direction_boundary) * temperature_step
                # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
                next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
                total_reward += reward
                session.report({'iterations':epoch,'total_reward':total_reward})
                # STORING THIS NEW TRANSITION INTO THE MEMORY
                dqn_model.remember([current_state, action, reward, next_state], game_over)
                # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
                inputs, targets = dqn_model.get_batch(model, batch_size = batch_size)
                # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
                loss += model.train_on_batch(inputs, targets)
                timestep += 1
                current_state = next_state
            # PRINTING THE TRAINING RESULTS FOR EACH EPOCH
            print("\n")
            print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
            print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
            print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
            
            # EARLY STOPPING
            if (early_stopping):
                if (total_reward <= best_total_reward):
                    patience_count += 1
                elif (total_reward > best_total_reward):
                    best_total_reward = total_reward
                    patience_count = 0
                if (patience_count >= patience):
                    print("Early Stopping")
                    break
            

# %%
config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-4, 1e-1),
    "layers": tune.choice([[32, 20], [64, 32], [64, 64],[128, 32], [128, 64], [128, 128]]),
    "dropout": tune.uniform(0.1, 0.5),
    "activation": tune.choice(['relu', 'tanh', 'sigmoid']),
    'optimizer': tune.choice(['adam', 'sgd', 'rmsprop', 'adamw']),
    'discount_factor': tune.choice([0.7, 0.8, 0.9, 0.95, 0.99]),
}

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)

resources_per_trial = {"cpu": 8, "gpu": 1}
scheduler = AsyncHyperBandScheduler(max_t=100, grace_period=3, reduction_factor=2)

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["total_reward", "training_iteration"],
)
analysis = tune.run(
    train_model,
    resources_per_trial=resources_per_trial,
    metric="total_reward",
    mode="max",
    config=config,
    num_samples=100,
    search_alg=algo,
    scheduler=scheduler,
    name="reinforce_optuna",
    local_dir="~/ray_results",
    progress_reporter=reporter,
    verbose=1,
)

print('Best hyperparameters found were: ', analysis.best_config)
# save it to a file
with open('best_hyperparameters_real.txt', 'w') as f:
    f.write(str(analysis.best_config))

# %%



