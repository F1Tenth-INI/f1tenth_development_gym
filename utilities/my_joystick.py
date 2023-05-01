"""
joystick race controller based on xbox bluetooth controller.
Xbox has 11 buttons.
buttons ABXY are first four buttons 0-3, then menu and window buttons are 4th and 5th from end, i.e. 7 and 6
"""
import logging
from typing import Tuple, Optional
import pygame # conda install -c cogsci pygame; maybe because it only is supplied for earlier python, might need conda install -c evindunn pygame ; sudo apt-get install libsdl-ttf2.0-0
from pygame import joystick
import platform
import time
from types import SimpleNamespace

INACTIVITY_RECONNECT_TIME = 15
RECONNECT_TIMEOUT = 5

LOGGING_LEVEL = logging.INFO  # set the overall default leval, change with --log option

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def my_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger


# Joystick connectivity
CHECK_FOR_JOYSTICK_INTERVAL = 100  # check for missing joystick every this many cycles
JOYSTICK_NUMBER = 0  # in case multiple joysticks, use this to set the desired one, starts from zero
JOYSTICK_STEERING_GAIN = .2  # gain for joystick steering input. 1 is too much for most people, 0.15 is recommended

logger = my_logger(__name__)
# logger.setLevel(logging.DEBUG)

# structure to hold driver control input
class car_command:
    """
    Car control commands from software agent or human driver, i.e., the throttle, steering, and brake input.

    Also includes reverse and autodrive_enabled boolean flags.
    """

    def __init__(self):
        self.steering=0  # value bounded by -1:1, this is the desired steering angle relative to maximum value and it is only the desired steering angle; actual steering angle is controlled by hidden dynamics of steering actuation and its limits
        self.throttle=0  # bounded to 0-1 from 0 to maximum possible, acts on car longitudinal acceleration according to hidden car and its motor dynamics
        self.brake=0  # bounded from 0-1
        self.reverse=False # boolean reverse gear
        self.autodrive_enabled = False # boolean activate or deactivate the autonomous driving, mapped to A key or Y Xbox controller button

    def __str__(self):
        try:
            s = 'steering={:.2f}, throttle={:.2f}, brake={:.2f} reverse={} auto={}'.format(self.steering, self.throttle, self.brake, self.reverse, self.autodrive_enabled)
        except TypeError:
            s = 'car command contains None!'
        return s


# user input, like quit, show statistics, etc.
# Separate from car_command which is just for controlling car

class user_input():
    """
    User input to l2race client.

    Includes the restart_car, restart_client, record_data, run_client_model, and quit commands
    """
    def __init__(self):
        self.choose_new_track = False
        self.restart_car=False # in debugging mode, restarts car at starting line
        self.restart_client=False # abort current run (server went down?) and restart from scratch
        self.quit=False # quit input from controller, mapped to ESC for keyboard and menu button for xbox controller
        self.run_client_model=False # run the client model of car
        self.toggle_recording=False # record data
        self.open_playback_recording=False
        self.close_playback_recording=False
        self.toggle_paused=False


def printhelp():
    print('\n-----------------------------------\nJoystick commands:\n'
          'steer with left joystick left|right\n'
          'throttle is right paddle, brake is left paddle\n'
          'B activates reverse gear\n'
          'Y activates autordrive control (if implemented)\n'
          'A runs the client ghost car model\n'
          'Menu button (to left of XYBA buttons) resets car\n'
          'X restarts client from scratch (if server went down)\n'
          'Windows button (by left joystick) quits\n-----------------------------------\n'
          )


class my_joystick:
    """"
    The read() method gets joystick input to return (car_command, user_input)
    """
    XBOX_ONE_BLUETOOTH_JOYSTICK = 'Xbox One S Controller' # XBox One when connected as Bluetooth in Windows
    XBOX_360_WIRELESS_RECEIVER = 'Xbox 360 Wireless Receiver' # XBox 360 when connected via USB wireless adapter on Linux
    XBOX_WIRED = 'Controller (Xbox One For Windows)' #Although in the name the word 'Wireless' appears, the controller is wired
    XBOX_ELITE = 'Xbox One Elite Controller' # XBox One when plugged into USB in Windows
    PS4_DUALSHOCK4 = 'Sony Interactive Entertainment Wireless Controller' # Only wired connection tested
    PS4_WIRELESS_CONTROLLER = 'Sony Computer Entertainment Wireless Controller' # Only wired connection on macOS tested

    def __init__(self, joystick_number=JOYSTICK_NUMBER):
        """
        Makes a new joystick instance.

        :param joystick_number: use if there is more than one joystick
        :returns: the instance

        :raises RuntimeWarning if no joystick is found or it is unknown type
        """
        self.joy: Optional[joystick.Joystick] = None
        self.numAxes: Optional[int] = None
        self.numButtons: Optional[int] = None
        self.axes = None
        self.buttons = None
        self.name: Optional[str] = None
        self.joystick_number: int = joystick_number
        self.lastTime = 0
        self.lastActive = 0
        self.run_user_model_pressed = False  # only used to log changes to/from running user model

        self._rev_was_pressed = False  # to go to reverse mode or toggle out of it
        self._auto_was_pressed = False  # to toggle out of autodrive if we pressed Y on joy to enter it
        joystick.init()
        count = joystick.get_count()
        if count < 1:
            raise RuntimeWarning('no joystick(s) found')

        self.platform = platform.system()

        self.lastActive=time.time()
        try:
            self.joy = joystick.Joystick(self.joystick_number)
        except:
            if self.platform == 'Linux': # Linux list joysticks from 3 to 0 instead of 0 to 3
                if self.joystick_number == 0:
                    self.joy = joystick.Joystick(3)
                else:
                    self.joy = joystick.Joystick(4 - self.joystick_number)

        self.joy.init()
        self.numAxes = self.joy.get_numaxes()
        self.numButtons = self.joy.get_numbuttons()
        self.name=self.joy.get_name()
        logger.info(f'joystick is named "{self.name}"')
        if not self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK \
                and not self.name == my_joystick.XBOX_360_WIRELESS_RECEIVER \
                and not self.name == my_joystick.XBOX_ELITE \
                and not self.name == my_joystick.XBOX_WIRED \
                and not self.name == my_joystick.PS4_WIRELESS_CONTROLLER \
                and not self.name == my_joystick.PS4_DUALSHOCK4:
            logger.warning('Name: {}'.format(self.name))
            logger.warning('Unknown joystick type {} found.'
                           'Add code to correctly map inputs by running my_joystick as main'.format(self.name))
            raise RuntimeWarning('unknown joystick type "{}" found'.format(self.name))
        logger.debug('joystick named "{}" found with {} axes and {} buttons'.format(self.name, self.numAxes, self.numButtons))

        # Joystick controls
        # Buttons A B X Y
        self.A = None  # ghost (client_model)
        self.B = None  # reverse
        self.X = None  # restart game
        self.Y = None  # autodrive
        self.O = None  # reverse
        self.Square = None  # restart game
        self.Triangle = None  # autodrive
        # Quit and Restart Client buttons
        self.Quit = None  # quit client
        self.uit = None  # quit client
        self.Restart_Client = None  # restart client
        # Analog Buttons and Axes
        self.Steering = None
        self.Throttle = None
        self.Brake = None


    def _connect(self):
        """
        Tries to connect to joystick

        :return: None

        :raises RuntimeWarning if there is no joystick
        """


    def check_if_connected(self):
        """
        Periodically checks if joystick is still there if there has not been any input for a while.
        If we have not read any input for INACTIVITY_RECONNECT_TIME and we have not checked for RECONNECT_TIMEOUT, then quit() the joystick and reconstruct it.

        :raises RuntimeWarning if check fails
        """
        now = time.time()
        if now - self.lastActive > INACTIVITY_RECONNECT_TIME and now - self.lastTime > RECONNECT_TIMEOUT:
            self.lastTime = now
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                return
            else:
               raise RuntimeWarning('no joystick found')

    def assign_buttons(self):

        if self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK and self.platform == 'Darwin':
            # Buttons A B X Y
            self.A = 0  # ghost (client_model)
            self.B = 1  # reverse
            self.X = 3  # restart game
            self.Y = 4  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 16  # quit client
            self.Restart_Client = 11  # restart client
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 4
            self.Brake = 5

        elif self.name == my_joystick.XBOX_ONE_BLUETOOTH_JOYSTICK:
            self.A = 0  # ghost (client_model)
            self.B = 1  # reverse
            self.X = 2  # restart game
            self.Y = 3  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 6  # quit client
            self.Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 5
            self.Brake = 4

        elif self.name == my_joystick.XBOX_360_WIRELESS_RECEIVER:
            # Buttons A B X Y
            self.A = 0  # ghost (client_model)
            self.B = 1  # reverse
            self.X = 2  # restart game
            self.Y = 3  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 6  # quit client
            self.Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 5
            self.Brake = 2

        elif self.name == my_joystick.XBOX_ELITE: # antonio's older joystick? also XBox One when plugged into USB cable on windows
            # Buttons A B X Y
            self.A = 0  # ghost (client_model)
            self.B = 1  # reverse
            self.X = 2  # restart game
            self.Y = 3  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 6  # quit client
            self.Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 5
            self.Brake = 2

        elif self.name == my_joystick.XBOX_WIRED:
            # Buttons A B X Y
            self.A = 0  # ghost (client_model)
            self.B = 1  # reverse
            self.X = 2  # restart game
            self.Y = 3  # autodrive
            # Quit and Restart Client buttons
            self.uit = 6  # quit client
            self.Restart_Client = 7  # restart client
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 5
            self.Brake = 4

        elif self.name == my_joystick.PS4_DUALSHOCK4:
            # Buttons X O Square Triangle
            self.X = 0  # ghost (client_model)
            self.O = 1  # reverse
            self.Square = 3  # restart game
            self.Triangle = 2  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 10
            self.Restart_Client = 3
            # Analog Buttons and Axes
            self.Steering = 0
            self.Throttle = 5
            self.Brake = 2

        elif self.name == my_joystick.PS4_WIRELESS_CONTROLLER:
            # Buttons X O Square Triangle
            self.X = 1  # ghost (client_model)
            self.O = 2  # reverse
            self.Square = 0  # restart game
            self.Triangle = 3  # autodrive
            # Quit and Restart Client buttons
            self.Quit = 8
            self.Restart_Client = 9
            # Analog Buttons and Axes
            self.Steering = 2
            self.Throttle = 4
            self.Brake = 3


    def read(self,  car_command:car_command,user_input:user_input):
        """
        Returns the car_command, user_input tuple. Use check_if_connected() and connect() to check and connect to joystick.

        :param car_command: the command to fill in
        :param user_input: the user input (e.g. open file) to fill in

        :raises RuntimeWarning if joystick disappears
        """

        self.check_if_connected()

        self.assign_buttons()

        if 'Xbox' in self.name:
            # Buttons A B X Y
            if self.joy.get_button(self.X)==1: user_input.restart_client = True  # X button - restart
            car_command.reverse = True if self.joy.get_button(self.B) == 1 else False  # B button - reverse
            if not car_command.reverse:  # only if not in reverse
                if self.joy.get_button(self.Y) == 1:  car_command.autodrive_enabled = True; self._auto_was_pressed=True  # Y button - autodrive
                elif self.joy.get_button(self.Y)==0 and self._auto_was_pressed: car_command.autodrive_enabled=False; self._auto_was_pressed=False
            user_input.run_client_model = self.joy.get_button(self.A)  # A button - ghost

        elif 'Sony' in self.name:
            # Buttons X O Square Triangle
            if  self.joy.get_button(self.Square)==1: user_input.restart_client = True # Square button - restart
            car_command.reverse = True if self.joy.get_button(self.O) == 1 else False  # O button - reverse
            if self.joy.get_button(self.Triangle) == 1:  car_command.autodrive_enabled = True; self._auto_was_pressed=True  # Y button - autodrive
            elif self.joy.get_button(self.Triangle)==0 and self._auto_was_pressed: car_command.autodrive_enabled=False; self._auto_was_pressed=False
            user_input.run_client_model = self.joy.get_button(self.X)  # X button - ghost

        # Quit and Restart Client buttons
        if self.joy.get_button(self.Restart_Client)==1: user_input.restart_car = True   # menu button - Restart Client
        if self.joy.get_button(self.Quit): user_input.quit = True # windows button - Quit Client

        # Analog Buttons and Axes
        car_command.steering = JOYSTICK_STEERING_GAIN * self.joy.get_axis(self.Steering)
        # Steering returns + for right push, which should make steering angle positive, i.e. CW
        car_command.throttle = (1 + self.joy.get_axis(self.Throttle)) / 2.  # Throttle
        car_command.brake = (1 + self.joy.get_axis(self.Brake)) / 2.  # Brake

        self.lastActive=time.time()


        logger.debug(self)

    def read_simple(self):

        self.check_if_connected()

        self.assign_buttons()

        # Analog Buttons and Axes
        steering = JOYSTICK_STEERING_GAIN * self.joy.get_axis(self.Steering)
        # Steering returns + for right push, which should make steering angle positive, i.e. CW
        throttle = (1 + self.joy.get_axis(self.Throttle)) / 2.  # Throttle

        return steering, throttle


# if __name__ == '__main__':
#     pygame.init()
#     joystick = my_joystick()
#     while True:
#         try:
#             pygame.event.get()
#             angular_control, translational_control = joystick.read_simple()
#             print('angular_control: {}'.format(angular_control))
#             print('translational_control: {}'.format(translational_control))
#             print('***********')
#         except pygame.error:
#             print('Joystick not detected')
#
#         pygame.time.wait(300)


import numpy as np
class joystick_simple:
    def __init__(self):
        pygame.init()
        self.joystick = my_joystick()

        self.axes = np.zeros(self.joystick.numAxes, dtype=float)

        self.angular_control_normed = None
        self.translational_control_normed = None


    def run_test(self):

        from colorama import init, deinit, Fore, Style
        init()


        old_axes = self.axes.copy()
        while True:
            pygame.event.get()  # must call get() to handle internal queue
            for i in range(self.joystick.numAxes):
                self.axes[i] = self.joystick.joy.get_axis(i)  # assemble list of analog values

            diff = self.axes - old_axes
            old_axes = self.axes.copy()

            # format output so changed are red
            axStr = 'axes: '
            axStrIdx = 'axes:'
            for i in range(self.joystick.numAxes):
                if abs(diff[i]) > 0.3:
                    axStr += (Fore.RED + Style.BRIGHT + '{:5.2f} '.format(self.axes[i]) + Fore.RESET + Style.DIM)
                    axStrIdx += '__' + str(i) + '__'
                else:
                    axStr += (Fore.RESET + Style.DIM + '{:5.2f} '.format(self.axes[i]))
                    axStrIdx += '______'

            print(axStr)
            pygame.time.wait(300)

    def read(self):
        
        pygame.event.get()  # must call get() to handle internal queue
        for i in range(self.joystick.numAxes):
            self.axes[i] = self.joystick.joy.get_axis(i)  # assemble list of analog values

        self.angular_control_normed = -self.axes[2]
        self.translational_control_normed = -self.axes[1]

        return self.angular_control_normed, self.translational_control_normed

    def read_test(self):
        while True:
            self.read()
            print("Angular control: {}".format(self.angular_control_normed))
            print("Translational control {}".format(self.translational_control_normed))
            print("***************")
        
        
            
if __name__ == '__main__':
    test = joystick_simple()
    test.read_test()
#
# if __name__ == '__main__':
#     from colorama import init, deinit, Fore, Style
#     import numpy as np
#     import atexit
#     init()
#     atexit.register(deinit)
#     pygame.init()
#     joy = my_joystick()
#     axes = np.zeros(joy.numAxes,dtype=float)
#     old_axes = axes.copy()
#     it = 0
#     print('Name of the current joystick is {}.'.format(joy.name))
#     print('Your platform name is {}'.format(joy.platform))
#     while True:
#         # joy.read()
#         # print("steer={:4.1f} throttle={:4.1f} brake={:4.1f}".format(joy.steering, joy.throttle, joy.brake))
#         pygame.event.get()  # must call get() to handle internal queue
#         for i in range(joy.numAxes):
#             axes[i] = joy.joy.get_axis(i)  # assemble list of analog values
#         diff = axes-old_axes
#         old_axes = axes.copy()
#         joy.buttons = list()
#         for i in range(joy.numButtons):
#             joy.buttons.append(joy.joy.get_button(i))
#
#         # format output so changed are red
#         axStr = 'axes: '
#         axStrIdx = 'axes:'
#         for i in range(joy.numAxes):
#             if abs(diff[i]) > 0.3:
#                 axStr += (Fore.RED+Style.BRIGHT+'{:5.2f} '.format(axes[i])+Fore.RESET+Style.DIM)
#                 axStrIdx += '__'+str(i)+'__'
#             else:
#                 axStr += (Fore.RESET+Style.DIM+'{:5.2f} '.format(axes[i]))
#                 axStrIdx += '______'
#
#         butStr = 'buttons: '
#         for button_index, button_on in enumerate(joy.buttons):
#             butStr = butStr+(str(button_index) if button_on else '_')
#
#         print(str(it) + ': ' + axStr + ' '+butStr)
#         print(str(it) + ': ' + axStrIdx)
#         it += 1
#         pygame.time.wait(300)
#
#
