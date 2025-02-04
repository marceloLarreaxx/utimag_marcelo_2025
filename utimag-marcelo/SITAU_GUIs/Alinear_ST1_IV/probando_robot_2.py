import time

import robot_helpers as rh
import numpy as np

IP_ROBOT = '192.168.2.11'
IP_PC = '192.168.2.13'

robot = rh.InterpreterHelper(IP_ROBOT, IP_PC)
robot.start_interpreter_mode()
robot.connect()
robot.start_listening()

home = np.array([-0.75*np.cos(np.pi/4), -0.75*np.cos(np.pi/4), 0.3, 0, 0, -np.pi/4])

# x = home[0] * 1000
# y = home[1] * 1000
# z = 300.0

# x = home[0]
# y = home[1]
# z = home[2]

# pose00 = [x, y, z, 0, 0, -np.pi / 4]
#
# pose0 = [x, y, z, 0, 0, -np.pi / 4]
# pose0 = [x, y, z, 0, 0, np.degrees(-np.pi / 4)]

# pose0[0:3] = [i/1000 for i in pose0[0:3]]

# pose = [x, y, z, 20, 0, np.degrees(-np.pi / 4)]
# pose = [x, y, z, np.radians(20), 0, -np.pi / 4]
# pose_trans = robot.get_pos_trans_str(pose0,pose)
# pose_trans = pose_trans[2:-1]
# comp = pose_trans.split(',')
# pose_trans2 = [float(c) for c in comp]


# pos1 = [0.2, 0.5, 0.1, 1.57,0,0]
# pos2 = [0.2, 0.5, 0.6, 1.57,0,0]

