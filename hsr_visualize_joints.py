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
STATE_NAMES = JOINT_NAMES

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



def visualize_joints_multiple_episodes(qpos_dict, joint_names=None, plot_path=None, ylim=None):
    """
    Visualize joint positions for multiple episodes.

    Args:
        qpos_dict (dict): Dictionary where keys are episode indices and values are qpos arrays.
        joint_names (list): List of joint names to visualize. If None, all joints are visualized.
        plot_path (str): Path to save the plot.
        ylim (tuple): Tuple specifying y-axis limits.
    """
    all_names = [name for name in STATE_NAMES]

    # Get indices of specified joints
    if joint_names:
        joint_indices = []
        for joint in joint_names:
            if joint in all_names:
                joint_indices.append(all_names.index(joint))
            else:
                print(f"Warning: Joint name '{joint}' not found in STATE_NAMES.")
        selected_names = [all_names[idx] for idx in joint_indices]
    else:
        joint_indices = range(len(all_names))
        selected_names = all_names

    # Debug output
    print(f"Selected joint indices: {joint_indices}")
    print(f"Selected joint names: {selected_names}")

    num_figs = len(joint_indices)
    if num_figs == 0:
        print("No valid joints selected for visualization. Exiting.")
        return

    # 横幅を広げるため、figsizeの横幅（w）を大きく設定
    h, w = 2, 10  # 横幅を10に設定
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # If only one joint is selected, axs might not be an array
    if num_figs == 1:
        axs = [axs]

    episode_labels_added = False  # Track if legend for episodes is already added

    for episode_idx, qpos in qpos_dict.items():
        qpos = np.array(qpos)

        # Validate qpos dimensions
        if qpos.shape[1] < len(all_names):
            print(f"Warning: Episode {episode_idx} has fewer dimensions ({qpos.shape[1]}) than expected ({len(all_names)}). Skipping.")
            continue

        for i, dim_idx in enumerate(joint_indices):
            ax = axs[i]
            ax.plot(qpos[:, dim_idx], label=f'Episode {episode_idx}' if not episode_labels_added else None)
        episode_labels_added = True  # Avoid duplicating episode labels in the legend

    for i, dim_idx in enumerate(joint_indices):
        ax = axs[i]
        ax.set_title(f'Joint {dim_idx}: {selected_names[i]}')
        if ylim:
            ax.set_ylim(ylim)

    # Add a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(qpos_dict), fontsize='small')

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches='tight')  # Ensure everything is saved, including legend outside plot
        print(f'Saved comparison plot to: {plot_path}')
    plt.close()


def visualize_joints_multiple_episodes(qpos_dict, joint_names=None, plot_path=None, ylim=None):
    """
    Visualize joint positions for multiple episodes.

    Args:
        qpos_dict (dict): Dictionary where keys are episode indices and values are qpos arrays.
        joint_names (list): List of joint names to visualize. If None, all joints are visualized.
        plot_path (str): Path to save the plot.
        ylim (tuple): Tuple specifying y-axis limits.
    """
    all_names = [name for name in STATE_NAMES]

    # Get indices of specified joints
    if joint_names:
        joint_indices = []
        for joint in joint_names:
            if joint in all_names:
                joint_indices.append(all_names.index(joint))
            else:
                print(f"Warning: Joint name '{joint}' not found in STATE_NAMES.")
        selected_names = [all_names[idx] for idx in joint_indices]
    else:
        joint_indices = range(len(all_names))
        selected_names = all_names

    # Debug output
    print(f"Selected joint indices: {joint_indices}")
    print(f"Selected joint names: {selected_names}")

    num_figs = len(joint_indices)
    if num_figs == 0:
        print("No valid joints selected for visualization. Exiting.")
        return

    # 横幅を広げるため、figsizeの横幅（w）を大きく設定
    h, w = 2, 10  # 横幅を10に設定
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # If only one joint is selected, axs might not be an array
    if num_figs == 1:
        axs = [axs]

    for episode_idx, qpos in qpos_dict.items():
        qpos = np.array(qpos)

        # Validate qpos dimensions
        if qpos.shape[1] < len(all_names):
            print(f"Warning: Episode {episode_idx} has fewer dimensions ({qpos.shape[1]}) than expected ({len(all_names)}). Skipping.")
            continue

        for i, dim_idx in enumerate(joint_indices):
            ax = axs[i]
            ax.plot(qpos[:, dim_idx], label=f'Episode {episode_idx}')  # ラベルをエピソードごとに付ける

    for i, dim_idx in enumerate(joint_indices):
        ax = axs[i]
        ax.set_title(f'Joint {dim_idx}: {selected_names[i]}')
        if ylim:
            ax.set_ylim(ylim)

    # グラフ全体に共通の legend を設定
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(len(qpos_dict), 5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # legend の分だけ上部のスペースを確保
    if plot_path:
        plt.savefig(plot_path, bbox_inches='tight')  # Ensure everything is saved, including legend outside plot
        print(f'Saved comparison plot to: {plot_path}')
    plt.close()


def main(args):
    dataset_dir = args['dataset_dir']
    start_episode_idx = args.get('start_episode_idx', 0)
    end_episode_idx = args.get('end_episode_idx', start_episode_idx + 1)  # Default to one episode if not specified
    joint_names = args.get('joint_names', None)

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
    compare_plot_path = os.path.join(dataset_dir, f'comparison_1222_odoms_{start_episode_idx}_to_{end_episode_idx - 1}.png')
    visualize_joints_multiple_episodes(qpos_dict, joint_names=joint_names, plot_path=compare_plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--start_episode_idx', action='store', type=int, help='Start episode index.', required=True)
    parser.add_argument('--end_episode_idx', action='store', type=int, help='End episode index.', required=False)
    parser.add_argument('--joint_names', nargs='+', type=str, help='List of joint names to visualize.', required=False)
    main(vars(parser.parse_args()))
