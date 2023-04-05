import numpy as np


class Environment(object):
    # Introducing and initializing all the parameters in and variables in the environment
    def __init__(self, optimal_temperature=(64.4, 75.2), initial_month=0, initial_number_users=10,
                 initial_rate_data=60):
        self.monthly_atmospheric_temperatures = [42.7, 46.65, 54.3, 61.65, 69.75, 76.8, 80, 78.9, 73.3, 62.85, 53.4,
                                                 45.4]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -17.0
        self.max_temperature = 112.0
        self.max_number_users = 100
        self.min_number_users = 10
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_naive = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_naive = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # Update the environment right after AI plays an action
    def update_env(self, direction, energy_ai, month):
        # get the reward

        # compute the energy spent by the server cooling system when there is no AI
        energy_naive = 0
        if self.temperature_naive < self.optimal_temperature[0]:
            energy_naive = self.optimal_temperature[0] - self.temperature_naive
            self.temperature_naive = self.optimal_temperature[0]
        elif self.temperature_naive > self.optimal_temperature[1]:
            energy_naive = self.temperature_naive - self.optimal_temperature[1]
            self.temperature_naive = self.optimal_temperature[1]
        # computing the reward
        self.reward = energy_naive - energy_ai
        # scaling the reward
        self.reward = 1e-3 * self.reward  # 1e-3 is a scaling factor - the maximum value of reward to get is 1. So,
        # the factor of 1e-2 is good enough, so we use 1e-3 to get better scaling so maximum reward is 0.1 now.
        # getting the next state

        # updating the atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        # updating the number of users
        self.current_number_users = self.current_number_users + \
                                    np.random.randint(-self.max_update_users, self.max_update_users)
        if self.current_number_users > self.max_number_users:
            self.current_number_users = self.max_number_users
        elif self.current_number_users < self.min_number_users:
            self.current_number_users = self.min_number_users
        # updating the data center energy consumption
        self.current_rate_data = self.current_rate_data + \
                                    np.random.randint(-self.max_update_data, self.max_update_data)
        if self.current_rate_data > self.max_rate_data:
            self.current_rate_data = self.max_rate_data
        elif self.current_rate_data < self.min_rate_data:
            self.current_rate_data = self.min_rate_data
