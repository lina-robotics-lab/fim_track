#!/usr/bin/env python3
import rospy
import sys
from std_msgs.msg import Float32
class virtual_light(object):
	"""
	Remember to use virtual_light.py (not just virtual_light) to refer to this package.

	The virtual light source used in simulations. 

	It simply publishes the raw light strength(the parameter k in our measurement model) under a target's namespace. 

	The default raw light strength is set to be constant 1.0
	"""
	def __init__(self, namespace, publish_rate=50):
		self.namespace=namespace
		self.raw_light_strength=1.0 
		rospy.init_node('virtual light {}'.format(namespace),anonymous=False)
		self.pub=rospy.Publisher('{}/raw_light_strength'.format(namespace),Float32,queue_size=10)
		self.rate=rospy.Rate(publish_rate)

	def emit(self):
		self.pub.publish(self.raw_light_strength)

if __name__ == '__main__':
	if len(sys.argv)<=1:
		print('Please specify the namespace of the virtual light!')
	else:
		namespace=sys.argv[1]
		v=virtual_light(namespace)
		while not rospy.is_shutdown():
			try:
				v.emit()
				v.rate.sleep()
			except rospy.exceptions.ROSInterruptException:
				# After Ctrl+C is pressed.
				pass
