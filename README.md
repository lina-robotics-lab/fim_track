# fim_track
The ROS utilities for fim tracking running on remote PC

---

__spin_and_collect__

**Prerequisite:** make sure the robot is publishing light sensor readings to the 'sensor_readings' topic,  in particular, by running the PYTHON3_publish_all_readings.py in light_sensor package. Also, bringup the robot so that we can publish to cmd_vel.

**Usage**: ``rosrun fim_track spin_and_collect.py robot_namespace total_time``

**robot_namespace** is either "" or "robotname". Notice the forward slash do not need to be included. 

**total_time** is in seconds

**Behavior**: If the script is directly run, it spins the robot for total_time seconds, at the same time collects light sensor data published to the "/robotname/sensor_readings", or ''sensor_readings'', then save all the recorded readings to "light_readings_robotname.txt" **in pwd(NOT in the src folder)**.

If you want to use the spin_and_collect class as a utility, refer to the following snippet.

```python
from spin_and_collect import spin_and_collect
awake_freq=10
robot_namespace='turtlebot1' # Can also be 'turtlebot3', etc.
total_time=30 # in seconds


sc=spin_and_collect(awake_freq)
sc.init_node()	
sc.spin_and_collect(robot_namespace,total_time)

print(np.array(sc.reading_records))
```

---

__simple_collect__

``rosrun fim_track spin_and_collect.py robot_namespace total_time``

The simplified version of spin_and_collect without spinning the robot, but only collect the sensor reading and store it in "light_readings_robotname.txt" **in pwd**. Can be used in parallel with manipulation packages like teleop.

**total_time** is an optional argument. By default it is np.inf.

```python
sc=spin_and_collect(awake_freq)
sc.init_node()	
sc.simple_collect(robot_namespace,total_time)

print(np.array(sc.reading_records))
```



---

__sensor_reading_listener__

**Prerequisite:** make sure the robot is publishing light sensor readings to the 'sensor_readings' topic,  in particular, by running the PYTHON3_publish_all_readings.py in light_sensor package. 

**Usage:** ``rosrun fim_track sensor_reading_listener.py robot_namespace``

**robot_namespace** is either "" or "robotname". Notice the forward slash do not need to be included. 

**Behavior:** collects light sensor data published to the "/robotname/sensor_readings", or ''sensor_readings'', and print it to console.

___

__manual_teleop_key__

This is a copy of the turtlebot3_teleop_key script which accept an additional namespace argument.

**Prerequisite**: the robot is brought up.

**Usage**: ``rosrun fim_track manual_teleop_key.py robot_namespace``

**robot_namespace** is either "" or "robotname". Notice the forward slash do not need to be included. 

**Behavior:** control the robot movement of the corresponding namespace with keyboard.

---

__calibrate_meas_coef__

This script collects target and robot location data from optitrack and robot light-sensor data in a synchronized way.

**Prerequisite:** The optitrack system is up and streaming location data to vrpn_client_node. The light sensors are publishing readings to sensor_readings.

**Usage:** ``rosrun fim_track calibrate_meas_coef.py robot_name_space target_namespace``. Then bringup the robot and ``rosrun fim_track manual_teleop_key.py robot_name_space`` to move the robot around while collecting data. Press Ctrl+C to end the recording and store the data to .txt files.

**robot_namespace is compulsory, ** and should be both a robot name and a rigid body name in the optitrack. Notice the forward slash do not need to be included. 

**target_namespace is optional.** By default it is "Lamp".

**Behavior:**  Press Ctrl+C to end the recording and store the data to .txt files. The data is recorded separately into three txt files: light_readings_turtlebotname.txt, robot_loc_turtlebotname.txt, target_loc_targetname.txt, and can be loaded using np.loadtxt(). All files contain the same number of data rows, and each row corresponds to the data collected at the same time.



---

__location_estimation__

This script does location estimation based on the locations and readings from the mobile sensors.

**Prerequisite:** The robots or simulated robots are brought up and are publishing to /mobile_sensor_x/sensor_readings and /mobile_sensor_x/sensor_coefs. This is normally done by bringing up actual Turtlebots properly or running the simulation launcher scripts in fim_track.

**Usage: ** Specify the following parameters in the location_estimation.py file:

- robot_names=['mobile_sensor_0',...]
- target_name='target_xx'
- qhint=np.array([x,y]), the initial guess of the target location to feed to the estimator.
	

Then run``'rosrun fim_track location_estimation.py``.

The estimated locations of the target calculated by all available algorithms will be published to the topics /location_estimation/algorithm_name.

---

__gazebo_simulation_launch__

This script contains utility to launch a Gazebo world with mobile sensors and targets at specified location. It is essentially the more versatile version of a launch file realized by a Python script. 

**Usage:**

It can be used as a stand-alone script, by running

```python
	rosrun fim_track gazebo_simulation_launch.py
```

Then follow the input prompt to set the noise level of Gaussian noises to be added to the virtual sensor readings. 

**Remark on setting noise level:** Usually a noise level<=0.01 is a reasonable one, where the location estimation algorithms can handle them properly. With noise level >=0.1, the location estimation algorithms would give very bad estimations, and the resulting target tracking will become very unstable.

And an empty Gazebo world with one target and three sensors will be launched. One may view the namespaces of the corresponding objects in rqt_graph, and control their movement via teleop/publishing to cmd_vel.

The ``launch_simulation()`` function in the file can also be used as a utility. 

```python
launch_simulation(sensor_poses=[],target_poses=[],basis_launch_file=None)	
```


The poses are lists of 4-vectors, each pose is in the format of: [x,y,z,Yaw]. 

The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.

The basis launch file usually contains information about in which .world to the simulation. If basis_launch file is not provided, then the empty world of gazebo_ros will be launched.

---

__turtlesim_launch__

This script contains utility to launch a turtlesim world. It is a much lighter simulation environment compared to gazebo, but implements the same dynamics of two-wheel Turtlebots.

**Usage:**

```python
	rosrun fim_track turtlesim_launch.py
```

Then follow the input prompt to set the noise level of Gaussian noises to be added to the virtual sensor readings. 

**Remark on setting noise level:** Usually a noise level<=0.01 is a reasonable one, where the location estimation algorithms can handle them properly. With noise level >=0.1, the location estimation algorithms would give very bad estimations, and the resulting target tracking will become very unstable.

By default, a turtlesim world with 3 mobile sensors and 1 moving target, in the character of turtles, will be launched. The namespaces and topics are exactly the same as launched that gazebo_simulation_launch.py

---
__monitor_robot_path__

This script records the trajectory of objects on ROS and visualize them in 2D live plots.

**Usage:**

First launch Gazebo simulation and spawn the mobile sensors and targets. Then run

```python
	rosrun fim_track monitor_robot_path.py
```

**Behavior:** a matplotlib window will pop up showing the current positions of the objects. Use teleop to move the objects so as to see the live plot getting updated.

---

__target_tracking_launch__

Launch the target tracking suite of nodes: location_estimation, multi_robot_controller, and single_robot_controller. 

These utility nodes will subscribe to mobile sensor's pose information as well as their sensor readings, then publish control signals to the cmd_vel topics of the mobile sensors, controlling them to move and track the target.

**Pre-requisite:** the mobile sensors are brought up and are publishing to pose, sensor_readings, and sensor_coefs topics. This is usually done by bringing up the actual Turtlebots or by running the simulation launcher scripts in fim_track.

**Usage:**

``rosrun fim_track target_tracking_launch.py``

The first input prompt will ask you to select the environment in which the experiment is running. Most commonly we will select g for Gazebo simulation or t for actual Turtlebots.

 The second prompt will ask you to input the correct number of mobile sensor we used in the experiment. The script will assume their namespaces are in the form of mobile_sensor_x, x=0:num_sensor-1.

After that, the target tracking suite will run and start to participate in the experiment.

---

__multi_robot_controller__

To be cont'd

---

__single_robot_controller__

To be cont'd