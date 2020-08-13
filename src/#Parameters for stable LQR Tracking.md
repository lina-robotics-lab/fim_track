# Parameters for a stable LQR Tracking

Q = [[2, 0, 0],
	 [0, 2, 0],
	 [0, 0, 1]]
R = [[10, 0],
	 [0, 1]]



Maximum Linear Velocity for Spline-based reference motion generation: 0.8 * 0.22. Keep maximum angular velocity unchanged.

A slower linear velocity in reference motion significantly slows down the actual speed of the robot.

Planning_dt for Spline-based reference motion generation: 1s.

Spinning/Wake-up frequecy for single_robot_controller: 10 Hz.

Maximum Linear Velocity for Waypoints Generation with Gradient Descent on trace(FIM^-1): 0.8 * 0.22. 

Planning_dt for Waypoints Generation: 1s. Which is kept consistent with the planning_dt in reference motion generation.

A larger planning_dt results in larger gaps between waypoints, allowing the LQR to focus more in merging with later part of the trajectory rather than aggressively correcting for the deviation from the initial waypoints. This again makes the path tracking more stable.

Spinning/Wake-up frequency for multi_robot_controller: 10 Hz. Which is consistent with single_robot_controller to ensure an abundant supply for new waypoints.

