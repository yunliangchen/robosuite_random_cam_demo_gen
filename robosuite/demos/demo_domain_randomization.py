"""
Script to showcase domain randomization functionality.
"""

import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

# We'll use instance randomization so that entire geom groups are randomized together
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        hard_reset=False,  # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
    )
    print(type(env))
    breakpoint()
    env = DomainRandomizationWrapper(env, randomize_color=False, 
                                     randomize_camera=True, 
                                     randomize_lighting=True,
                                     camera_randomization_args={'camera_names': env.sim.model.camera_names.remove("eye_in_hand"), 
                                                                'fovy_perturbation_size': 25.0, 
                                                                'position_perturbation_size': 0.2, 
                                                                'randomize_fovy': True, 
                                                                'randomize_position': True, 
                                                                'randomize_rotation': True, 
                                                                'rotation_perturbation_size': 0.15}, 
                                     lighting_randomization_args={'ambient_perturbation_size': 0.1, 
                                                                  'diffuse_perturbation_size': 0.1, 
                                                                  'direction_perturbation_size': 0.5, 
                                                                  'light_names': None, 
                                                                  'position_perturbation_size': 0.5, 
                                                                  'randomize_active': True, 
                                                                  'randomize_ambient': False, 
                                                                  'randomize_diffuse': False, 
                                                                  'randomize_direction': True, 
                                                                  'randomize_position': True, 
                                                                  'randomize_specular': True, 
                                                                  'specular_perturbation_size': 0.5}
                                    )
    env.reset()
    env.viewer.set_camera(camera_id=2)

    # camera_wrapper = CameraWrapper(env)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(300):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        # Robot base in the world coord is usually but not always -0.56 0.0 0.912
        # base position
        robot_base_pos = np.array([float(x) for x in env.robots[0].robot_model._elements["root_body"].get("pos").split(" ")])
        camera_pos = env.camera_modder.get_pos("frontview")
        camera_pos = np.array(camera_pos) - robot_base_pos # the actual camera position if the robot base is at 0, 0, 0
        camera_quat = T.convert_quat(env.camera_modder.get_quat("frontview"), to="xyzw")
        camera_rot = T.quat2mat(camera_quat)
        # robosuite camera is right, up, backward (opengl), but we want right, down, forward (opencv)
        camera_rot[:, 1] = -camera_rot[:, 1]
        camera_rot[:, 2] = -camera_rot[:, 2]
        camera_quat = T.mat2quat(camera_rot)

        camera_fovy = env.camera_modder.get_fovy("frontview")

        camera_pose_wrt_robot = T.make_pose(camera_pos, T.quat2mat(camera_quat))
        print("Camera pose in the world frame:", camera_pose_wrt_robot)

        env.render()
        import time
        time.sleep(0.03)
