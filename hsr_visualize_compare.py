import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed



JOINT_NAMES = ["odom_x", "odom_y", "odom_z", "arm_lift_joint", "arm_flex_joint", "arm_roll_joint",'wrist_flex_joint','wrist_roll_joint','hand_motor_joint']
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        # qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    # return qpos, qvel, action, image_dict
    return qpos, action, image_dict
def visualize_joints_multiple_episodes(qpos_dict, plot_path=None, ylim=None):
    """
    Visualize joint positions for multiple episodes.

    Args:
        qpos_dict (dict): Dictionary where keys are episode indices and values are qpos arrays.
        plot_path (str): Path to save the plot.
        ylim (tuple): Tuple specifying y-axis limits.
    """
    num_dim = qpos_dict[next(iter(qpos_dict))].shape[1]  # Assuming all episodes have the same joint dimension.
    all_names = [name for name in STATE_NAMES]
    num_figs = num_dim
    h, w = 2, num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    for episode_idx, qpos in qpos_dict.items():
        qpos = np.array(qpos)
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(qpos[:, dim_idx], label=f'Episode {episode_idx}')

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f'Saved comparison plot to: {plot_path}')
    plt.close()

def main(args):
    dataset_dir = args['dataset_dir']
    start_episode_idx = args.get('start_episode_idx', 0)
    end_episode_idx = args.get('end_episode_idx', start_episode_idx + 1)  # Default to one episode if not specified

    qpos_dict = {}
    for episode_idx in range(start_episode_idx, end_episode_idx):
        dataset_name = f'episode_{episode_idx}'
        try:
            qpos, action, image_dict = load_hdf5(dataset_dir, dataset_name)
            qpos_dict[episode_idx] = qpos
        except Exception as e:
            print(f"Failed to load episode {episode_idx}: {e}")
            continue

    if not qpos_dict:
        print("No episodes were loaded. Please check the dataset directory and episode indices.")
        return

    # Visualize joint comparison for all selected episodes
    compare_plot_path = os.path.join(dataset_dir, f'comparison_{start_episode_idx}_to_{end_episode_idx - 1}.png')
    visualize_joints_multiple_episodes(qpos_dict, plot_path=compare_plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--start_episode_idx', action='store', type=int, help='Start episode index.', required=True)
    parser.add_argument('--end_episode_idx', action='store', type=int, help='End episode index.', required=False)
    main(vars(parser.parse_args()))
