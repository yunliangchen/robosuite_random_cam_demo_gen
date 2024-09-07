"""
Convenience script to tune a camera view in a mujoco environment.
Allows keyboard presses to move a camera around in the viewer, and
then prints the final position and quaternion you should set
for your camera in the mujoco XML file.
"""

import argparse
import time
import xml.etree.ElementTree as ET

import numpy as np
from pynput.keyboard import Controller, Key, Listener

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover
from robosuite.utils.mjcf_utils import find_elements, find_parent
from scipy.spatial.transform import Rotation

# some settings
DELTA_POS_KEY_PRESS = 0.05  # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1  # delta camera angle per key press

# 5 levels, 20 cameras each
# radius levels: N(1.0, 0.1), N(1.0, 0.2), N(1.1, 0.3), N(1.1, 0.4), N(1.2, 0.5)
# theta levels: U(50, 60), U(45, 65), U(40, 70), U(40, 70), U(35, 75)
# phi levels: U(-pi/12, pi/12), U(-pi/10, pi/10), U(-pi/8, pi/8), U(-pi/6, pi/6), U(-pi/4, pi/4)

radius_mean_levels = [1.0, 1.0, 1.1, 1.1, 1.2]
radius_std_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
theta_min_levels = [50, 45, 40, 40, 35]
theta_max_levels = [60, 65, 70, 70, 75]
phi_min_levels = [-np.pi/12, -np.pi/10, -np.pi/8, -np.pi/6, -np.pi/4]
phi_max_levels = [np.pi/12, np.pi/10, np.pi/8, np.pi/6, np.pi/4]
noise_std_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

def sample_half_hemisphere(num_samples, radius_mean=1.2, radius_std=0.5, theta_min=30, theta_max=75, phi_min=-np.pi/4, phi_max=np.pi/4):
    radius = np.random.normal(radius_mean, radius_std, num_samples)
    hemisphere_center = np.array([0, 0, 0])
    # theta = np.random.uniform(np.pi/5, np.pi/2.2, num_samples)  # Angle with respect to the z-axis
    theta = np.random.uniform(theta_min/180*np.pi, theta_max/180*np.pi, num_samples)  # Angle with respect to the z-axis
    # phi = np.random.uniform(-np.pi*3.7/4, np.pi*3.7/4, num_samples)  # Azimuthal angle
    phi = np.random.uniform(phi_min, phi_max, num_samples)  # Azimuthal angle
    positions = np.zeros((num_samples, 3))
    positions[:, 0] = radius * np.sin(theta) * np.cos(phi)  # x-coordinate
    positions[:, 1] = radius * np.sin(theta) * np.sin(phi)  # y-coordinate
    positions[:, 2] = radius * np.cos(theta)  # z-coordinate

    # Calculate orientations (quaternions)
    backward_directions = positions - hemisphere_center
    backward_directions /= np.linalg.norm(backward_directions, axis=1, keepdims=True)
    right_directions = np.cross(np.tile(np.array([0, 0, 1]), (num_samples, 1)), backward_directions)  # Assuming right direction is along the x-axis
    right_directions /= np.linalg.norm(right_directions, axis=1, keepdims=True)
    up_directions = np.cross(backward_directions, right_directions)
    up_directions /= np.linalg.norm(up_directions, axis=1, keepdims=True)

    rotations = np.array([np.column_stack((right, down, forward)) for right, down, forward in zip(right_directions, up_directions, backward_directions)])
    # rotations = np.array([                                                        [[ 0.   ,       0.70614784, -0.70806442     ],
    #                                                         [ 1.    ,      0.      ,    0.                ],
    #                                                         [ 0.     ,    -0.70806442 ,-0.70614784     ]]])

    # Convert rotation matrices to quaternions
    quaternions = []
    for rotation_matrix in rotations:
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        quaternions.append(quaternion)

    quaternions = np.array(quaternions)


    return positions, quaternions



class KeyboardHandler:
    def __init__(self, camera_mover):
        """
        Store internal state here.

        Args:
            camera_mover (CameraMover): Playback camera class
        cam_body_id (int): id corresponding to parent body of camera element
        """
        self.camera_mover = camera_mover

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    def on_press(self, key):
        """
        Key handler for key presses.

        Args:
            key (int): keycode corresponding to the key that was pressed
        """

        try:
            # controls for moving rotation
            if key == Key.up:
                # rotate up
                self.camera_mover.rotate_camera(point=None, axis=[1.0, 0.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.down:
                # rotate down
                self.camera_mover.rotate_camera(point=None, axis=[-1.0, 0.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.left:
                # rotate left
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 1.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.right:
                # rotate right
                self.camera_mover.rotate_camera(point=None, axis=[0.0, -1.0, 0.0], angle=DELTA_ROT_KEY_PRESS)

            # controls for moving position
            elif key.char == "w":
                # move forward
                self.camera_mover.move_camera(direction=[0.0, 0.0, -1.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "s":
                # move backward
                self.camera_mover.move_camera(direction=[0.0, 0.0, 1.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "a":
                # move left
                self.camera_mover.move_camera(direction=[-1.0, 0.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "d":
                # move right
                self.camera_mover.move_camera(direction=[1.0, 0.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "r":
                # move up
                self.camera_mover.move_camera(direction=[0.0, 1.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "f":
                # move down
                self.camera_mover.move_camera(direction=[0.0, -1.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == ".":
                # rotate counterclockwise
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 0.0, 1.0], angle=DELTA_ROT_KEY_PRESS)
            elif key.char == "/":
                # rotate clockwise
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 0.0, -1.0], angle=DELTA_ROT_KEY_PRESS)

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.

        Args:
            key: [NOT USED]
        """
        pass


def print_command(char, info):
    """
    Prints out the command + relevant info entered by user

    Args:
        char (str): Command entered
        info (str): Any additional info to print
    """
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    args = parser.parse_args()

    print("\nWelcome to the camera tuning script! You will be able to tune a camera view")
    print("by moving it around using your keyboard. The controls are printed below.")

    print("")
    print_command("Keys", "Command")
    print_command("w-s", "zoom the camera in/out")
    print_command("a-d", "pan the camera left/right")
    print_command("r-f", "pan the camera up/down")
    print_command("arrow keys", "rotate the camera to change view direction")
    print_command(".-/", "rotate the camera view without changing view direction")
    print("")

    # read camera XML tag from user input
    inp = input(
        "\nPlease paste a camera name below \n"
        "OR xml tag below (e.g. <camera ... />) \n"
        "OR leave blank for an example:\n"
    )

    if len(inp) == 0:
        if args.env != "Lift":
            raise Exception("ERROR: env must be Lift to run default example.")
        print("\nUsing an example tag corresponding to the frontview camera.")
        print("This xml tag was copied from robosuite/models/assets/arenas/table_arena.xml")
        inp = '<camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>'

    # remember the tag and infer some properties
    from_tag = "<" in inp
    notify_str = (
        "NOTE: using the following xml tag:\n"
        if from_tag
        else "NOTE: using the following camera (initialized at default sim location)\n"
    )

    print(notify_str)
    print("{}\n".format(inp))

    cam_tree = ET.fromstring(inp) if from_tag else ET.Element("camera", attrib={"name": inp})
    CAMERA_NAME = cam_tree.get("name")

    # make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()

    # Create the camera mover
    camera_mover = CameraMover(
        env=env,
        camera=CAMERA_NAME,
    )

    # Make sure we're using the camera that we're modifying
    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # Infer initial camera pose
    if from_tag:
        initial_file_camera_pos = np.array(cam_tree.get("pos").split(" ")).astype(float)
        initial_file_camera_quat = T.convert_quat(np.array(cam_tree.get("quat").split(" ")).astype(float), to="xyzw")
        # Set these values as well
        camera_mover.set_camera_pose(pos=initial_file_camera_pos, quat=initial_file_camera_quat)
        # Optionally set fov if specified
        cam_fov = cam_tree.get("fovy", None)
        if cam_fov is not None:
            env.sim.model.cam_fovy[camera_id] = float(cam_fov)
    else:
        initial_file_camera_pos, initial_file_camera_quat = camera_mover.get_camera_pose()
    # Define initial file camera pose
    initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

    # remember difference between camera pose in initial tag and absolute camera pose in world
    initial_world_camera_pos, initial_world_camera_quat = camera_mover.get_camera_pose()
    initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(camera_mover=camera_mover)


    curriculum_level = 4
    cam_positions, cam_quaternions = sample_half_hemisphere(1000, radius_mean_levels[curriculum_level], radius_std_levels[curriculum_level], theta_min_levels[curriculum_level], theta_max_levels[curriculum_level], phi_min_levels[curriculum_level], phi_max_levels[curriculum_level]) # sample 1000 camera positions

    # just spin to let user interact with window
    spin_count = 0
    i = 0
    pose_collections = []
    j = 0
    while j < 20:
        if spin_count % 50 == 0:
            print("spin_count: ", spin_count, "set camera")
            # robot base: -0.56 0.0 0.912
            camera_mover.set_camera_pose(pos=cam_positions[i] + np.array([0, 0.0, 0.912]) +np.random.normal(0, noise_std_levels[curriculum_level], 3), quat=cam_quaternions[i])
            i += 1

        action = np.zeros(env.action_dim)
        obs, reward, done, _ = env.step(action)
        env.render()
        spin_count += 1

        # ask user whether it is good
        if (spin_count-1) % 50 == 0:
            print(spin_count)
            inp = input("Is this camera view good? (y/n): ")
            if inp.lower() == "y":
                

            # if spin_count % 50 == 0:
                # convert from world coordinates to file coordinates (xml subtree)
                camera_pos, camera_quat = camera_mover.get_camera_pose()
                world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
                file_camera_pose = world_in_file.dot(world_camera_pose)
                # TODO: Figure out why numba causes black screen of death (specifically, during mat2pose --> mat2quat call below)
                camera_pos, camera_quat = T.mat2pose(file_camera_pose)
                camera_quat = T.convert_quat(camera_quat, to="wxyz")

                # print("\n\ncurrent camera tag you should copy")
                cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
                cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
                cam_tree.set("name", "agentview_{}".format(j+81))
                print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))

                pose_collections.append(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))
                j += 1
            else:
                print("not saved")
                continue
    # save pose_collections to a txt file
    with open("/home/lawchen/project/robosuite/robosuite/models/assets/arenas/camera_poses_curriculum.txt", "w") as f:
        for pose in pose_collections:
            f.write(pose + "\n")