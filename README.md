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