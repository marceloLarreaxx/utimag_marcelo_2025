import logging
import re
import socket
import numpy as np
from scipy.spatial.transform import Rotation
import time

"""
Este código es para utilizar el robot en modo interprete. Para comunicarse con el robot se usan tres sockets. 

1. UR_INIT_INTERPRETER_MODE_PORT: es solo para enviar el comando que inicia el modo intérprete del controlador
del robot. En ese modo el robot espera comando a traves del UR_INTERPRETER_SOCKET
2. UR_INTERPRETER_SOCKET: por aquí recibe comandos/programas. En este caso le enviamos "threads" que quedan 
ejecutandose en paralelo en el controlador.
3. LISTEN_PORT: acá python recibe los mensajes que envía el robot, con las coordenadas o lo que sea.
"""

UR_INIT_INTERPRETER_MODE_PORT = 30001
UR_INTERPRETER_SOCKET = 30020
LISTEN_PORT = 30001

# WARNING: definir en funcion del TCP offset por defecto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
coef_home = 725
HOME_POSE = np.array([-coef_home*np.cos(np.pi/4), -coef_home*np.cos(np.pi/4), 300, 0, 0, 45])
# HOME_POSE = np.array([-0.7*np.cos(np.pi/4), -0.7*np.cos(np.pi/4), 0.121, np.pi, 0, 0.0])
MAX_FORCE = 60


class InterpreterHelper:
    """
    Basado en:
    https://www.universal-robots.com/articles/ur/programming/interpreter-mode/
    """

    log = logging.getLogger("interpreter.InterpreterHelper")
    STATE_REPLY_PATTERN = re.compile(r"(\w+):\W+(\d+)?")

    def __init__(self, ip_robot, ip_pc, interpreter_port=UR_INTERPRETER_SOCKET,
                 init_interpreter_mode_port=UR_INIT_INTERPRETER_MODE_PORT, listen_port=LISTEN_PORT, home_pose=HOME_POSE):

        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.init_interpreter_mode_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip_robot = ip_robot
        self.ip_pc = ip_pc
        self.interpreter_port = interpreter_port
        self.init_interpreter_mode_port = init_interpreter_mode_port
        self.listen_port = listen_port
        self.home = home_pose
        self.init_interpreter_mode_socket.connect((self.ip_robot, self.init_interpreter_mode_port))

    def start_interpreter_mode(self):
        print('starting interpreter')
        self.init_interpreter_mode_socket.send(b'interpreter_mode()\n')
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.connect((self.ip_robot, self.init_interpreter_mode_port))
        #     s.send(b'interpreter_mode()\n')
        print('interpreter mode started')

    def start_listening(self, timeout=10):
        self.listen_socket.bind((self.ip_pc, self.listen_port))
        self.listen_socket.settimeout(timeout)
        self.listen_socket.listen(1)
        c = 'a=False while not a: a=socket_open("{}", {}) sleep(0.2) end\n'.format(self.ip_pc, self.listen_port)
        self.execute_command(c)
        conn, addr = self.listen_socket.accept()
        self.listen_conn = conn
        print(addr)

    def connect(self):
        try:
            self.control_socket.connect((self.ip_robot, self.interpreter_port))
            print('control socket connected')
            # ejecutar el thread para controlar las colisiones
            self.force_thread(MAX_FORCE)
        except socket.error as exc:
            self.log.error(f"socket error = {exc}")
            raise exc

    def reconnect(self):
        self.control_socket.close()  # cierro socket de envìo
        self.listen_socket.close()
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.start_interpreter_mode()
        self.connect()
        self.start_listening()

    def disconnect(self):
        self.end_interpreter()
        self.control_socket.close()
        self.listen_conn.close()
        self.listen_socket.close()
        self.init_interpreter_mode_socket.close()
        print('robot disconnected')

    def get_reply(self):
        """
        read one line from the socket
        :return: text until new line
        """
        collected = b''
        while True:
            part = self.control_socket.recv(1)
            if part != b"\n":
                collected += part
            elif part == b"\n":
                break
        return collected.decode("utf-8")

    def execute_command(self, command):
        """
        Send single line command to interpreter mode, and wait for reply
        :param command:
        :return: ack, or status id
        """
        self.log.debug(f"Command: '{command}'")
        if not command.endswith("\n"):
            command += "\n"

        self.control_socket.send(command.encode("utf-8"))

        # ARREGLAR!!!!!!!!!!!!!!!!!!!!!!!
        # raw_reply = self.get_reply() # corregir por si no hay respuesta
        # self.log.debug(f"Reply: '{raw_reply}'")
        # # parse reply, raise exception if command is discarded
        # reply = self.STATE_REPLY_PATTERN.match(raw_reply)
        # if reply.group(1) == "discard":
        #     print("Interpreter discarded message", raw_reply)
        # return reply.group(2)

    def clear(self):
        return self.execute_command("clear_interpreter()")

    def skip(self):
        return self.execute_command("skipbuffer")

    def abort_move(self):
        return self.execute_command("abort")

    def get_last_interpreted_id(self):
        return self.execute_command("statelastinterpreted")

    def get_last_executed_id(self):
        return self.execute_command("statelastexecuted")

    def get_last_cleared_id(self):
        return self.execute_command("statelastcleared")

    def get_unexecuted_count(self):
        return self.execute_command("stateunexecuted")

    def end_interpreter(self):
        return self.execute_command("end_interpreter()")

    def get_actual_tcp_pose(self):
        # str1 = 'popup(" + ','.join(map(str, tcp_pose)) + '])'
        # self.execute_command('popup("here i am",title="Title")')
        self.execute_command('socket_send_string(to_str(get_actual_tcp_pose()))')
        # print(self.listen_conn.recv(128).decode())
        return format_pose_string(self.listen_conn.recv(128).decode())

    def get_tcp_offset(self):
        self.execute_command('socket_send_string(to_str(get_tcp_offset()))')
        return format_pose_string(self.listen_conn.recv(128).decode())

    def go_home(self, t):
        str1 = self.return_rotvec_pose(self.home)
        # print(str1)
        trg_pose_str = 'movej({},t={})'.format(str1, t)
        print(trg_pose_str)
        self.execute_command(trg_pose_str)

    def move_to_pose(self,trg_pose,t, block=False):
        trg_pose = np.array(trg_pose,dtype=float)
        trg_pose[0:3] = trg_pose[0:3]/1000  # Cambio los tres primero elementos a metros
        # print(trg_pose)
        trg_pose_str = 'movej(p['+','.join(map(str,trg_pose))+'],'
        trg_pose_str = f"{trg_pose_str}t={t})"
        # print(trg_pose_str)
        self.execute_command(trg_pose_str)
        if block:
            time.sleep(t)

    def return_rotvec_pose(self,trg_pose):
        '''SE LE PASA UNA POSE EN mm Y ÀNGULOS DE EULER EN GRADOS Y DEVUELVE LA MISMA EM m Y VECTOR ROTACIÓN EN RADIANES'''
        trg_pose = np.array(trg_pose,dtype=float)
        trg_pose[0:3] = trg_pose[0:3]/1000  # Cambio los tres primeros elementos a metros
        r = Rotation.from_euler('xyz', trg_pose[3:], degrees=True)
        euler_rot = r.as_rotvec()
        trg_pose2 = np.concatenate((trg_pose[0:3],np.round(euler_rot,2)))
        trg_pose_str = 'p['+','.join(map(str,trg_pose2))+']'
        # print(trg_pose_str)
        return trg_pose_str
        # self.execute_command(trg_pose_str)

    def set_tcp_offset(self,tcp_pose):
        tcp_pose_str = self.return_rotvec_pose(tcp_pose)
        tcp_pose_str = 'set_tcp(' + tcp_pose_str + ')'
        self.execute_command(tcp_pose_str)

    def force_thread(self, max_force):
        self.execute_command(ur_threads['contact_test'].format(max_force))
        # ejecuta el contact_test thread
        self.execute_command('force_thrd = run contact_test()')

    def move_from_home(self,trg_pose,t, block=False):
        str1 = self.return_rotvec_pose(trg_pose)
        # print(str1)
        # str2 = 'p[' + ','.join(map(str, self.home)) + ']'
        str2 = self.return_rotvec_pose(self.home)
        # print(str2)
        str3 = 'movej(pose_trans({},{}),t={})'.format(str2,str1,t)
        # print(str3)
        self.execute_command(str3)
        if block:
            time.sleep(t)

    def move_from_home1(self, pose0, trg_pose, t, block=False):
        str1 = self.return_rotvec_pose(trg_pose)
        # print(str1)
        str2 = self.return_rotvec_pose(pose0)
        # str2 = 'p[' + ','.join(map(str, pose0)) + ']'
        # str2 = self.return_rotvec_pose(pose0)
        # print(str2)
        str3 = 'movel(pose_trans({},{}),t={})'.format(str2,str1,t)
        self.execute_command(str3)
        if block:
            time.sleep(t)

    def get_pos_trans_str(self, pose1, pose2):
        str1 = 'p{}'.format(pose1)
        str2 = 'p{}'.format(pose2)
        str3 = 'pose_trans('+str1+', '+str2+')'
        str4 = 'socket_send_string(to_str('+str3+'))'
        self.execute_command(str4)
        return self.listen_conn.recv(128).decode()
        # return format_pose_string(self.listen_conn.recv(128).decode())

    def set_home(self,home):
        self.home = home

    def set_offset_tcp_0(self, tcp_pose0):
        tcp_pose_str = self.return_rotvec_pose(tcp_pose0)
        tcp_pose_str = 'set_tcp(' + tcp_pose_str + ')'
        self.execute_command(tcp_pose_str)

    def get_relative_pose_to_home(self):
        home_init = self.home
        home_rot = Rotation.from_euler('z', np.pi / 4)
        actual_tcp = self.get_actual_tcp_pose()
        actual_rot = Rotation.from_rotvec(actual_tcp[3:])
        rot_rel = home_rot.inv() * actual_rot
        rot_rel = rot_rel.as_euler('xyz', degrees=True)
        #########
        xyz_rel = np.array(actual_tcp[0:3]) - np.array(home_init[0:3])
        pose_rel = np.round(np.concatenate((xyz_rel, rot_rel)), 2)
        return xyz_rel, rot_rel

def format_pose_string(s, decimals=3):
    """ Esto es para cuando se usa get_actual_tcp_pose() en el robot, y se pasa el resultado
    como un string tipo 'p[0.45626, 0.234534, 0.34534, 2.3, 0, 1.9]. Se pasa las distancias
    de metros a mm, y se quita el exceso de decimales"""

    s = s[2:]  # quitar p[
    s = s[:-1]  # quitar ]
    float_strings = s.split(', ')
    factor = [1000, 1000, 1000, 1, 1, 1]
    nums = [np.around(k * float(x), decimals) for x, k in zip(float_strings, factor)]
    return nums


# p = plus, m = minus. Entonces: zp es z plus, zm es z minus
ur_threads = {'force_test': ('thread force_test(): while True: if force() > {}: textmsg("max force") halt end '
                             'sync() end return False end'),
              # puse force() porque con tool_contact() se paraba por nada...revisar todo
              'contact_test': ('thread contact_test(): while True: if force() > {}: textmsg("contact") '
                               'socket_send_string("contact halt") halt end sync() end return False end'),
              # traslaciones
              'xp': 'thread xp(): speedl([{}, {}, 0, 0, 0, 0], {}, {}) end',
              'xm': 'thread xm(): speedl([{}, {}, 0, 0, 0, 0], {}, {}) end',
              'yp': 'thread yp(): speedl([{}, {}, 0, 0, 0, 0], {}, {}) end',
              'ym': 'thread ym(): speedl([{}, {}, 0, 0, 0, 0], {}, {}) end',
              'zp': 'thread zp(): speedl([0, 0, {}, 0, 0, 0], {}, {}) end',
              'zm': 'thread zm(): speedl([0, 0, {}, 0, 0, 0], {}, {}) end',
              # rotaciones
              'rxp': 'thread rxp(): speedl([0, 0, 0, {}, {}, 0], {}, {}) end',
              'rxm': 'thread rxm(): speedl([0, 0, 0, {}, {}, 0], {}, {}) end',
              'ryp': 'thread ryp(): speedl([0, 0, 0, {}, {}, 0], {}, {}) end',
              'rym': 'thread rym(): speedl([0, 0, 0, {}, {}, 0], {}, {}) end',
              'rzp': 'thread rzp(): speedl([0, 0, 0, 0, 0, {}], {}, {}) end',
              'rzm': 'thread rzm(): speedl([0, 0, 0, 0, 0, {}], {}, {}) end'}
