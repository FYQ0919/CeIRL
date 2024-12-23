import numpy

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils import Logger
import numpy as np
import math
import torch 
import os 



def normalize_vector(vector):
    vectors = []
    for i in range(vector.shape[0]):
        magnitude = math.sqrt(sum([component ** 2 for component in vector[i]]))
        vector_xy = ([component / magnitude for component in vector[i]])
        vector_xy.append(0)
        vectors.append(vector_xy)

    return np.array(vectors)

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, current_value):
        # Calculate error
        error = self.setpoint - current_value

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error
        I = self.Ki * self.integral

        # Derivative term
        derivative = error - self.prev_error
        D = self.Kd * derivative

        # Save error for next derivative calculation
        self.prev_error = error

        # Calculate control output
        output = P + I + D

        return output

def play_policy(env_cfg,train_cfg, policy, env, cmd_vel = [1.5,0.0,0.0],robot_idx = 10, joint_index = 2, record = False, move_camera = False):
    logger = Logger(env.dt)
    robot_index = robot_idx # which robot is used for logging
    joint_index = joint_index # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    history_vel_error = numpy.zeros(100)
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    cmd_vel = cmd_vel
    env.set_eval()
    with torch.inference_mode():
        env.set_command(cmd_vel)
        env.reset()
    obs_dict = env.get_observations()  # TODO: check, is this correct on the first step?
    
    RECORD_FRAMES = record
    MOVE_CAMERA = move_camera   
    with torch.inference_mode():
        print(env.max_episode_length)
        for i in range(2*int(env.max_episode_length)):
            # print(i)
            actions = policy.act_inference(obs_dict)
            obs_dict, rewards, dones, infos= env.step(actions.detach())
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            vel_error = np.linalg.norm((env.base_lin_vel.cpu().numpy() - cmd_vel),axis=1)
            # print(env.base_lin_vel.cpu().numpy())
            mean = np.mean(vel_error)
            mean = mean[np.newaxis]
            # std = np.std(vel_error)
            history_vel_error = np.concatenate((history_vel_error[1:],mean))
            # print(f"mean_error {np.mean(history_vel_error)}")
            # print(f"std_error { np.std(history_vel_error)}")

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos': env.dof_pos[robot_index].cpu().numpy(),
                        'dof_pos_target': env.joint_pos_target[robot_index].cpu().numpy(),
                        'dof_vel': env.dof_vel[robot_index].cpu().numpy(),
                        'dof_torque': env.torques[robot_index].cpu().numpy(),
                        'command': env.commands[robot_index].cpu().numpy(),
                        'base_vel': env.base_lin_vel[robot_index].cpu().numpy(),
                        'base_quat': env.base_quat[robot_index].cpu().numpy(),
                        'base_ang_vel': env.base_ang_vel[robot_index].cpu().numpy(),
                        'foot_contact': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        'action': actions[robot_index].cpu().numpy(),
                    }
                )
            elif i==stop_state_log:
                logger.new_plot_states()
                print("plotting states")
            if  0 < i < stop_rew_log:
                if infos["train/episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["train/episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()


def play_move_target(env_cfg, train_cfg, policy, env, target_pos=[10, 5.0], robot_idx=10, joint_index=2, record=False,
                move_camera=False):
    logger = Logger(env.dt)
    robot_index = robot_idx  # which robot is used for logging
    joint_index = joint_index  # which joint is used for logging
    stop_state_log = 10  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    # cmd_vel = cmd_vel
    env.set_eval()
    with torch.inference_mode():
        # env.set_command(cmd_vel)
        env.reset()
    obs_dict = env.get_observations()  # TODO: check, is this correct on the first step?
    start_pos = obs_dict["base_pos"].detach().cpu()[:,:2]

    target_pos = np.array(target_pos)
    end_pos = start_pos + target_pos
    RECORD_FRAMES = record
    MOVE_CAMERA = move_camera

    # Initialize PID controller for x and y positions
    pid_controller_x = PID(Kp=1.0, Ki=0.0, Kd=0.05, setpoint=end_pos[:,0])
    pid_controller_y = PID(Kp=1.0, Ki=0.0, Kd=0.05, setpoint=end_pos[:,1])

    current_pos = obs_dict['base_pos'].detach().cpu()[:,:2]

    max_step = 0
    with torch.inference_mode():
        # for i in range(3 * int(env.max_episode_length)):
        while np.linalg.norm(end_pos[0] - current_pos[0,:2]) > 0.5 and max_step < 2000:
            max_step += 1
            # Get current position from observations (assuming it's in obs_dict)
            current_pos = obs_dict['base_pos'].detach().cpu()  # Adjust key according to actual observation structure
            print(current_pos[0,:2] - start_pos[0])
            # Calculate control actions using PID controllers
            control_action_x = pid_controller_x.step(current_pos[:,0])
            control_action_y = pid_controller_y.step(current_pos[:,1])
            control_action = np.stack([control_action_x, control_action_y],axis=1)
            control_cmd = normalize_vector(control_action)
            env.set_command(control_cmd)
            print(f"control command :{control_cmd[0]}")
            obs_dict = env.get_observations()

            actions = policy.act_inference(obs_dict)
            obs_dict, rewards, dones, infos = env.step(actions.detach())
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            # if i < stop_state_log:
            #     logger.log_states(
            #         {
            #             'dof_pos': env.dof_pos[robot_index].cpu().numpy(),
            #             'dof_pos_target': env.joint_pos_target[robot_index].cpu().numpy(),
            #             'dof_vel': env.dof_vel[robot_index].cpu().numpy(),
            #             'dof_torque': env.torques[robot_index].cpu().numpy(),
            #             'command': env.commands[robot_index].cpu().numpy(),
            #             'base_vel': env.base_lin_vel[robot_index].cpu().numpy(),
            #             'base_quat': env.base_quat[robot_index].cpu().numpy(),
            #             'base_ang_vel': env.base_ang_vel[robot_index].cpu().numpy(),
            #             'foot_contact': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
            #             'action': actions[robot_index].cpu().numpy(),
            #         }
            #     )
            # elif i == stop_state_log:
            #     logger.new_plot_states()
            #     print("plotting states")
            # if 0 < i < stop_rew_log:
            #     if infos["train/episode"]:
            #         num_episodes = torch.sum(env.reset_buf).item()
            #         if num_episodes > 0:
            #             logger.log_rewards(infos["train/episode"], num_episodes)
            # elif i == stop_rew_log:
            #     logger.print_rewards()