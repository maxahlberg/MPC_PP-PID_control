# MPC_PP-PID_control
MPC trajectory generation with Pure Pursuit and P control following.

This code takes waypoints as inputs and creates a track.
That track is then transformed from Global coordinates to Frenet.
A model predictive controller is tracking the centerline using a linear point mass model with constraints.
A pure pursuit and p controller is following the MPC trajectory one lap around the FSG comeptition track of 2017
The control inputs, Steering angle and Throttle is sent to a bicycle model representing the real vehicle. The state of the "real" car is then transformed into the Frent frame to feedback the state to the MPC


frenet_optimal_trajectory.py is the main file. First time I write in Python.
