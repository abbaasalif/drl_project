# Developed a Deep Q-learning based agent for HVAC system optimization

The environment is written using simplistic models based on Heat transfer equations
The agent demonstrates superior performance compared to regular thermostat based control in a simulated environment and the real environment data is collected using the CPU temperature of a system while running different benchmarks over a Raspberry Pi.

The repository presents a Deep Q-learning-based agent to optimize HVAC (Heating, Ventilation, and Air Conditioning) systems. The environment is modeled using fundamental heat transfer equations, enabling the agent to outperform traditional thermostat-based controls.

**Repository Structure:**

- **`brain.py` and `brain_pyt.py`**: Define the neural network architectures and related functions for the agent.
- **`dqn.py` and `dqn_pyt.py`**: Implement the Deep Q-Network algorithms.
- **`environment.py` and `real_env.py`**: Simulate the HVAC environment using heat transfer models.
- **`training.py` and `training_real.py`**: Scripts to train the agent in simulated and real environments, respectively.
- **`testing.py` and `testing_real.py`**: Evaluate the performance of the trained agent.
- **`train_hyper.py`, `train_hyper_real.py`, and `train_hyper.ipynb`**: Hyperparameter tuning scripts and notebook.
- **`regression_model.ipynb`**: Notebook detailing the development of a regression model related to the project.
- **Model and Scaler Files**:
  - `model.h5` and `model_real.h5`: Saved models for simulated and real environments.
  - `finalized_model.sav` and `finalized_scaler.sav`: Serialized model and scaler objects.
- **Log Files**:
  - `log_sensor.csv`: Contains sensor data logs.
- **Hyperparameter Files**:
  - `best_hyperparameters.txt` and `best_hyperparameters_real.txt`: Document optimal hyperparameters for respective environments.

**Getting Started:**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/abbaasalif/drl_project.git
   cd drl_project
   ```

2. **Install Dependencies:**
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is absent, manually install necessary packages such as TensorFlow, Keras, NumPy, and Pandas.*

3. **Train the Agent:**
   To train the agent in the simulated environment:
   ```bash
   python training.py
   ```
   For training in a real-world environment:
   ```bash
   python training_real.py
   ```

4. **Test the Agent:**
   After training, evaluate the agent's performance:
   ```bash
   python testing.py
   ```
   For real-world environment testing:
   ```bash
   python testing_real.py
   ```

**Hyperparameter Tuning:**

Utilize the provided scripts and notebook for hyperparameter optimization:

- **Scripts**: `train_hyper.py` and `train_hyper_real.py`
- **Notebook**: `train_hyper.ipynb`

**Additional Resources:**

- **Regression Model Development**: Refer to `regression_model.ipynb` for insights into the regression model associated with this project.
- **Sensor Data Logs**: Examine `log_sensor.csv` for recorded sensor data during experiments.

**Contributing:**

Contributions are welcome. Please fork the repository and submit a pull request with detailed information about your changes.

