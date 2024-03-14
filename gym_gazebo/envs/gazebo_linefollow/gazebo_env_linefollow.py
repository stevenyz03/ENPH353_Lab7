
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Converts data into an OpenCV image and displays it with state based on blue lines.
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Convert image to HSV color space to better identify blue colors
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the range of blue color in HSV
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply the blue mask to the bottom 25% of the image
        height, width = cv_image.shape[:2]
        bottom_image_masked = blue_mask[int(0.6 * height):height, :]

        # Find contours in the masked image
        contours, _ = cv2.findContours(bottom_image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initial state and done flag
        NUM_BINS = 3
        state = [0] * 10
        done = False

        if contours:
            # Assume the largest contour in the bottom 25% of the image is the line
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask for the largest contour
            mask = np.zeros_like(bottom_image_masked)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Segment the bottom image into 10 bins and analyze where the line is
            segments_with_contour = []
            for i in range(10):
                bin_slice = mask[:, i*mask.shape[1]//10:(i+1)*mask.shape[1]//10]
                if cv2.countNonZero(bin_slice) > 0:
                    segments_with_contour.append(i)
            
            # If there are any segments with the contour, update the middle one
            if segments_with_contour:
                middle_index = segments_with_contour[len(segments_with_contour) // 2]
                state[middle_index] = 1

            self.timeout = 0  # Reset timeout because line is detected
        else:
            self.timeout += 1
            
        # Check if line was not detected for more than 30 frames
        if self.timeout > 30:
            done = True
            self.timeout = 0  # Reset timeout for the next episode

        # Visualize the state array on the full image
        for i, bin_state in enumerate(state):
            color = (0, 255, 0) if bin_state == 1 else (0, 0, 255)
            cv2.rectangle(cv_image, (i * width // 10, 0), ((i+1) * width // 10, 50), color, -1)
            cv2.putText(cv_image, str(bin_state), (i * width // 10 + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Show the full image with the state array visualization
        cv2.imshow('window', cv_image)
        cv2.waitKey(1)

        return state, done


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.55
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

            # Calculate the reward based on the state array and the action
        line_position_weighted = sum([i * state[i] for i in range(len(state))]) / sum(state) if sum(state) > 0 else -1
        line_position_centered = (line_position_weighted - len(state) / 2) / (len(state) / 2) if line_position_weighted >= 0 else 0

        
        if line_position_weighted >= 0:
            # Positive rewards for correct direction, scaled by distance from center
            if action == 1 and line_position_centered < 0:  # LEFT turn when line is on the LEFT
                reward = 5 * abs(line_position_centered)
            elif action == 2 and line_position_centered > 0:  # RIGHT turn when line is on the RIGHT
                reward = 5 * abs(line_position_centered)
            elif abs(line_position_centered) <= 0.2 and action == 0:  # The line is centered and the action is FORWARD
                reward = 2  # High reward for going straight when the line is centered
            elif action == 0:  # FORWARD
                reward = 1  # Minimal reward for moving forward, regardless of line position
            else:
                # Negative rewards for incorrect direction, scaled by distance from center
                reward = -10 * abs(line_position_centered)
        else:
            # If no line is detected, penalize heavily
            reward = -200
            done = True

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
