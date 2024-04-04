from utilities.my_joystick import joystick_simple
import pygame
from Control_Toolkit_ASF.Controllers import template_planner

class manual_planner(template_planner):
    """
    Example Planner
    """

    def __init__(self):

        super().__init__()

        pygame.init()
        self.joystick = joystick_simple()

        self.angular_control = None
        self.translational_control = None

        self.angular_control_normed = None
        self.translational_control_normed = None

    def process_observation(self, ranges=None, ego_odom=None):
        self.angular_control_normed, self.translational_control_normed = self.joystick.read()

        self.translational_control = self.translational_control_normed
        self.angular_control = self.angular_control_normed/3

        return self.angular_control, self.translational_control
