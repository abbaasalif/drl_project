# Importing the libraries
import numpy as np
from pickle import load
# BUILDING THE ENVIRONMENT IN A CLASS

class Environment(object):
    
    # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE ENVIRONMENT
    
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_bytes_received=0, initial_cpu_percent=0):
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_bytes_received = 0 
        self.max_bytes_received = 141
        self.max_update_bytes = 10
        self.min_cpu_percent = 0.0 
        self.max_cpu_percent = 100
        self.max_update_cpu = 0.1
        self.initial_bytes_received = initial_bytes_received
        self.current_bytes_received = initial_bytes_received
        self.initial_cpu_percent = initial_cpu_percent
        self.current_cpu_percent = initial_cpu_percent
        # load the standard scaler and scale the data from finalized_scaled.sav
        self.sc = load(open('/home/ubuntu/drl_project/finalized_scaler.sav','rb'))
        # load the model and predict the data from finalized_model.sav
        self.model = load(open('/home/ubuntu/drl_project/finalized_model.sav','rb'))
        inputs = np.array([[self.current_bytes_received, self.current_cpu_percent, self.atmospheric_temperature]])
        inputs = self.sc.transform(inputs)
        self.intrinsic_temperature = self.model.predict(inputs)[0]
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # MAKING A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE AI PLAYS AN ACTION
    
    def update_env(self, direction, energy_ai, month):
        
        # GETTING THE REWARD
        
        # Computing the energy spent by the server's cooling system when there is no AI
        energy_noai = 0
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif (self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        # Computing the Reward
        self.reward = energy_noai - energy_ai
        # Scaling the Reward
        self.reward = 1e-3 * self.reward
        
        # GETTING THE NEXT STATE
        
        # Updating the atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        # Updating the cpu percent
        self.current_cpu_percent += np.random.uniform(-self.max_update_cpu, self.max_update_cpu)
        if (self.current_cpu_percent > self.max_cpu_percent):
            self.current_cpu_percent = self.max_cpu_percent
        elif (self.current_cpu_percent < self.min_cpu_percent):
            self.current_cpu_percent = self.min_cpu_percent
        # Updating the bytes received
        self.current_bytes_received += np.random.randint(-self.max_update_bytes, self.max_update_bytes)
        if (self.current_bytes_received > self.max_bytes_received):
            self.current_bytes_received = self.max_bytes_received
        elif (self.current_bytes_received < self.min_bytes_received):
            self.current_bytes_received = self.min_bytes_received
        # Computing the Delta of Intrinsic Temperature
        past_intrinsic_temperature = self.intrinsic_temperature
        inputs = np.array([[self.current_bytes_received, self.current_cpu_percent, self.atmospheric_temperature]]) 
        self.intrinsic_temperature = self.model.predict(self.sc.transform(inputs))[0]
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        # Computing the Delta of Temperature caused by the AI
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai
        # Updating the new Server's Temperature when there is the AI
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        # Updating the new Server's Temperature when there is no AI
        self.temperature_noai += delta_intrinsic_temperature
        
        # GETTING GAME OVER
        
        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
        
        # UPDATING THE SCORES
        
        # Updating the Total Energy spent by the AI
        self.total_energy_ai += energy_ai
        # Updating the Total Energy spent by the server's cooling system when there is no AI
        self.total_energy_noai += energy_noai
        
        # SCALING THE NEXT STATE
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_bytes_received = (self.current_bytes_received - self.min_bytes_received) / (self.max_bytes_received - self.min_bytes_received)
        scaled_cpu_percent = (self.current_cpu_percent - self.min_cpu_percent) / (self.max_cpu_percent - self.min_cpu_percent)
        next_state = np.matrix([scaled_bytes_received, scaled_cpu_percent, scaled_temperature_ai])
        
        # RETURNING THE NEXT STATE, THE REWARD, AND GAME OVER
        
        return next_state, self.reward, self.game_over

    # MAKING A METHOD THAT RESETS THE ENVIRONMENT
    
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_bytes_received = self.initial_bytes_received
        self.current_cpu_percent = self.initial_cpu_percent
        inputs = np.array([[self.current_bytes_received, self.current_cpu_percent, self.atmospheric_temperature]])
        inputs = self.sc.transform(inputs)
        self.intrinsic_temperature = self.model.predict(inputs)[0]
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # MAKING A METHOD THAT GIVES US AT ANY TIME THE CURRENT STATE, THE LAST REWARD AND WHETHER THE GAME IS OVER
    
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_bytes_received = (self.current_bytes_received - self.min_bytes_received) / (self.max_bytes_received - self.min_bytes_received)
        scaled_cpu_percent = (self.current_cpu_percent - self.min_cpu_percent) / (self.max_cpu_percent - self.min_cpu_percent)
        current_state = np.matrix([scaled_bytes_received, scaled_cpu_percent, scaled_temperature_ai])
        return current_state, self.reward, self.game_over
