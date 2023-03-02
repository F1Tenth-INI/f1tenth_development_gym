import os
def get_gym_path ():
    gym_path = ""
    abs_path = os.path.abspath(".")
    if ".ros" in abs_path:
        import rospkg
        rospack = rospkg.RosPack()
        path = rospack.get_path('autonomous_driving') #/home/racecar/catkin_ws/src/autonomous_driving
        gym_path = path+"/f1tenth_development_gym"
        
    return gym_path
        