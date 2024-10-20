import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define fuzzy variables for traffic density, time of day, and green light duration
traffic_density = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic_density')
time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')
green_light_duration = ctrl.Consequent(np.arange(0, 61, 1), 'green_light_duration')

# Define membership functions for traffic density (low, medium, high)
traffic_density['low'] = fuzz.trimf(traffic_density.universe, [0, 0, 50])
traffic_density['medium'] = fuzz.trimf(traffic_density.universe, [30, 50, 70])
traffic_density['high'] = fuzz.trimf(traffic_density.universe, [50, 100, 100])

# Define membership functions for time of day (non-peak, peak)
time_of_day['non_peak'] = fuzz.trimf(time_of_day.universe, [0, 0, 12])
time_of_day['peak'] = fuzz.trimf(time_of_day.universe, [10, 24, 24])

# Define membership functions for green light duration (short, moderate, long)
green_light_duration['short'] = fuzz.trimf(green_light_duration.universe, [0, 0, 20])
green_light_duration['moderate'] = fuzz.trimf(green_light_duration.universe, [15, 30, 45])
green_light_duration['long'] = fuzz.trimf(green_light_duration.universe, [40, 60, 60])

# Visualize the membership functions
traffic_density.view()
time_of_day.view()
green_light_duration.view()

# Define the rules for the fuzzy system
rule1 = ctrl.Rule(traffic_density['low'] & time_of_day['non_peak'], green_light_duration['short'])
rule2 = ctrl.Rule(traffic_density['low'] & time_of_day['peak'], green_light_duration['moderate'])
rule3 = ctrl.Rule(traffic_density['medium'] & time_of_day['non_peak'], green_light_duration['moderate'])
rule4 = ctrl.Rule(traffic_density['medium'] & time_of_day['peak'], green_light_duration['long'])
rule5 = ctrl.Rule(traffic_density['high'] & time_of_day['non_peak'], green_light_duration['long'])
rule6 = ctrl.Rule(traffic_density['high'] & time_of_day['peak'], green_light_duration['long'])

# Control system
green_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
green_light_sim = ctrl.ControlSystemSimulation(green_light_ctrl)

# Simulate the system for some input values (traffic density and time of day)
green_light_sim.input['traffic_density'] = 75  # High traffic
green_light_sim.input['time_of_day'] = 18  # Peak hours

# Compute the output based on the input values
green_light_sim.compute()

# Print and visualize the output
print(f"Recommended Green Light Duration: {green_light_sim.output['green_light_duration']} seconds")
green_light_duration.view(sim=green_light_sim)

# Show the plots
plt.show()

print('Deep Marathe - 53004230016')
