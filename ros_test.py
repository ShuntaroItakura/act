import math
import moveit_commander
import rospy
import tf
from ros_utils import *
rospy.init_node("arm")


whole_body.set_joint_value_target( [-0.5028328985367991, 0.5034089559146024, 1.424383472034283, 0.6874298758573738, -1.2222573390549234, 0.0008161939209926236, -1.9187516505033588, 0.1923553255357175, 0.0])
whole_body.go()
a = whole_body.get_current_joint_values()
print(a)