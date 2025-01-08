import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from hsr_constants import DT
from hsr_constants import PUPPET_GRIPPER_JOINT_OPEN
from hsr_utils import load_data, load_data_with_logging # data functions
from hsr_utils import sample_box_pose, sample_insertion_pose # robot functions
from hsr_utils import compute_dict_mean, set_seed, detach_dict # helper functions
from hsr_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

# from sim_env import BOX_POSE

# Gazebo関連のトピック
# import rospy
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64
# base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
# joint_state_pub = rospy.Publisher('/hsrb/joint_states', JointState, queue_size=10)

import math
import moveit_commander
import rospy
import tf
from ros_utils import *
# rospy.init_node("arm")


def initialize_ros_node():
    """Gazebo環境の初期化"""
    rospy.init_node("hsr_gazebo_eval")
    rospy.sleep(1)

# def move_robot_via_gazebo(action):
#     """Gazeboトピックを通してアクションをHSRに送信"""
#     joint_state_msg = JointState()
#     joint_state_msg.position = action
#     joint_state_pub.publish(joint_state_msg)

import IPython
e = IPython.embed

def main(args):
    # print(torch.cuda.is_available())  # Trueが期待される
    # print(torch.cuda.device_count()) 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set_seed(1)
    set_seed(8)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    action_save_dir_num = args['action_save_dir_num']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from hsr_constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.hsr_constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    #
    if is_sim:
        initialize_ros_node()
        # pass

    else:
        from aloha_scripts.hsr_constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 9
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'seed':args['seed']
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'action_seed': args['action_seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'action_save_dir_num':action_save_dir_num
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_epoch_4765_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()
    
    
    log_file_train = "train_shuffle_log.txt"
    log_file_val = "val_shuffle_log.txt"
    log_file_train = os.path.join(ckpt_dir, f"train_shuffle_log.txt")
    log_file_val = os.path.join(ckpt_dir, f"val_shuffle_log.txt")

    train_dataloader, val_dataloader, stats = load_data_with_logging(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, 
        log_file_train, log_file_val
    )

    # train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy= ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy



def make_policy_seed(policy_class, policy_config,seed):
    if policy_class == 'ACT':
        policy= ACTPolicy(policy_config,seed)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    rgbd = RGBD()   # RgbdCamera オブジェクトの初期化

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    # print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from hsr_sim_env import make_gazebo_env
        
        env = make_gazebo_env(task_name, rgbd)  # `rgbd` を `make_gazebo_env` に渡す
        
        # env_max_reward = env.task.max_reward
        # Gazebo環境用の最大報酬値を手動で設定
        if 'sim_transfer_cube' in task_name:
            env_max_reward = 4  # TransferCubeTaskの最大報酬値
        elif 'sim_insertion' in task_name:
            env_max_reward = 4  # InsertionTaskの最大報酬値
        elif 'sim_hsr_pick' in task_name:
            env_max_reward = 1  # ピックタスクの適切な報酬値
        else:
            env_max_reward = 0  # デフォルトの値

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # Gazeboトピックの初期化
    # make_gazebo_env(task_namec)
    z_output_dir = os.path.join(ckpt_dir, "latent_data")
    os.makedirs(z_output_dir, exist_ok=True)

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    #zを抽出
    mu_list = []
    logvar_list = []
    latent_z_list = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        print(rollout_id,'rollout_id')
        ### set task
        if 'sim_transfer_cube' in task_name:
            # BOX_POSE[0] = sample_box_pose() # used in sim reset
            continue
        elif 'sim_hsr_pick' in task_name:
            # BOX_POSE[0] = sample_box_pose()
            print('sim_hsr_pick')
            
        elif 'sim_insertion' in task_name:
            # BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
            continue
        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            act_i = True
            for t in range(max_timesteps):
                print(t,'t')
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)
                
                #

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                # print(obs,'obs')
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    # image_list.append({'main': obs['image']})
                    image_list.append({'hand_camera': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)
                # print(curr_image,curr_image.shape,'curr_image')
                print(qpos,'qpos')
                print(qpos.shape,'qpos.shape')

                ### query policy
                if config['policy_class'] == "ACT":
                    # 保存先ディレクトリとファイル名
                    action_save_dir = "/home/developer/workspace/act/action_gen"  # 保存先ディレクトリ (例: ./output)
                    action_file_name = f"actions{config['action_save_dir_num']}.txt"
                    action_file_name_json = f"actions{config['action_save_dir_num']}.json"

                    # ディレクトリが存在しない場合は作成
                    os.makedirs(action_save_dir, exist_ok=True)

                    # フルパスを作成
                    file_path = os.path.join(action_save_dir, action_file_name)
                    file_path_json = os.path.join(action_save_dir,action_file_name_json)

                    if t % query_frequency == 0:
                        # all_actions = policy(qpos, curr_image)
                        # print(all_actions,"all_actions")
                        #zを抽出
                        all_actions, mu,logvar = policy(qpos, curr_image)  # 潜在変数 z を取得
                        # latent_z_list.append(z.cpu().numpy())  # z を保存
                        idx = 0
                        for all_action in all_actions:
                            raw_action = all_action.squeeze(0).cpu().numpy()
                            action = post_process(raw_action)
                            # テキストファイルに書き込む
                            if act_i:
                                
                                with open(file_path, "a") as file:  # "a" は追記モード
                                    for action_idx in action.tolist():
                                        file.write(f"{config['action_seed']},{idx}:{action_idx[5:8]}\n")  # 1行に1つのアクションを記録
                                        idx =+1
                                import json
                                with open(file_path_json, "w") as file:
                                    json.dump(action.tolist(), file, indent=4)
                                
                        
                    if temporal_agg:
                        print('temporal_agg')
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # for all_action in all_actions:
                        #     raw_action = all_action.squeeze(0).cpu().numpy()
                        #     action = post_process(raw_action)
                            # テキストファイルに書き込む
                            # if act_i:
                            #     idx = 0
                            #     with open(file_path, "a") as file:  # "a" は追記モード
                            #         for action_idx in action.tolist():
                            #             file.write(f"{config['action_seed']},{idx}:{action_idx[6:8]}\n")  # 1行に1つのアクションを記録
                            #             idx =+1
                            #     import json
                            #     with open(file_path_json, "w") as file:
                            #         json.dump(action.tolist(), file, indent=4)
                            # # print(action)
                            #     act_i = False
                        raw_action = all_actions[:, t % query_frequency]
                        # raw_action = all_actions[:,9]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                print(raw_action,'raw_action')
                action = post_process(raw_action)
                target_qpos = action
                


                # whole_body.set_joint_value_target( target_qpos)
                # whole_body.go()
                # whole_body.get_current_joint_values()

                # print(target_qpos,'aaaa')



                ### step the environment
                # ts = env.step(target_qpos)
                # observation, reward, done = env.step(target_qpos)
                # print(reward,'reward')

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                # rewards.append(ts.reward)
                # rewards.append(reward)

            plt.close()
            # 各ロールアウト終了時に保存
            np.save(os.path.join(z_output_dir, f"mu_rollout_{rollout_id}.npy"), np.array(mu_list))
            np.save(os.path.join(z_output_dir, f"logvar_rollout_{rollout_id}.npy"), np.array(logvar_list))
            mu_list = []  # 次のロールアウトのためにリセット
            logvar_list = []

        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        # rewards = np.array(rewards)
        # episode_return = np.sum(rewards[rewards!=None])
        # episode_returns.append(episode_return)
        # episode_highest_reward = np.max(rewards)
        # highest_rewards.append(episode_highest_reward)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        print(len(image_list),'len_image_list')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    # # success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    # avg_return = np.mean(episode_returns)
    # # summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    # print(summary_str)

    # # save success rate to txt
    # result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    # with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
    #     f.write(summary_str)
    #     f.write(repr(episode_returns))
    #     f.write('\n\n')
    #     f.write(repr(highest_rewards))

    return None, None


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def extract_a_hat(loss_dict, policy):
    """
    ポリシーモデルの出力辞書から a_hat を抽出する関数。

    :param loss_dict: forward_pass から得られた辞書
    :param policy: 使用しているポリシーモデル
    :return: a_hat または None
    """
    if isinstance(policy, (ACTPolicy, CNNMLPPolicy)):
        return loss_dict.pop('a_hat', None)
    return None

def extract_z(loss_dict,policy):
    mu = loss_dict.pop('mu',None)
    logvar = loss_dict.pop('logvar',None)
    return(mu,logvar)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # チェックポイントの復元
    start_epoch = 0
    best_ckpt_info = None
    min_val_loss = np.inf

    #既存のチェックポイントがる場合、ロード
    # ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_7000_seed_0.ckpt')
    # if os.path.exists(ckpt_path):
    #     print(f"Loading checkpoint from {ckpt_path}")
    #     checkpoint = torch.load(ckpt_path)
    #     policy.load_state_dict(checkpoint)
    #     # optimizer.load_state_dict(checkpoint)
        # start_epoch = checkpoint['epoch'] + 1
        # min_val_loss = checkpoint['min_val_loss']
        # best_ckpt_info = checkpoint['best_ckpt_info']
        # print(f"Resumed training from epoch {start_epoch} with min_val_loss {min_val_loss:.6f}")


    z_output_dir = os.path.join(ckpt_dir, "latent_data")
    os.makedirs(z_output_dir, exist_ok=True)


    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                # print(f"Forward pass result: {forward_dict}")
                epoch_dicts.append(forward_dict)
            
            epoch_summary ,a_hat= compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)


         # 追加トレーニングフェーズ
        train_predictions = []
        train_ground_truth = []

        mu_list = []
        logvar_list = []

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # forward_dict,mu,logvar = forward_pass(data, policy)
            forward_dict = forward_pass(data, policy)

            # qpos_data, image_data, action_data, is_pad = data

            # a_hat,  is_pad_hat, (mu, logvar)= policy(qpos_data, image_data, action_data, is_pad)

            # #mu,logvarの保存
            # if epoch == num_epochs-1:
            #     print('num_epochs-1:',num_epochs-1)
            #     mu_list.append(mu.cpu().numpy())
            #     logvar_list.append(mu.cpu().numpy())

            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(f"Epoch {epoch}, Batch {batch_idx}, forward_dict: {forward_dict}")
            train_history.append(detach_dict(forward_dict))
        print(f"Epoch {epoch}, Batch {batch_idx}, train_history slice: {train_history[(batch_idx + 1) * epoch: (batch_idx + 1) * (epoch + 1)]}")

        epoch_summary ,a_hat= compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        action_data = data[2]  # 正解値を追加
        # print(action_data,'action_data')
        # print(a_hat,'a_hat')

        # 予測値と正解値を記録
        train_predictions.append(a_hat.detach().cpu().numpy())  # detachして記録
        train_ground_truth.append(action_data.detach().cpu().numpy())  # 正解値も記録
        # print(train_predictions,'train_predictions')
        # print(train_ground_truth,'train_ground_truth')

        save_step = 100
        if epoch % save_step == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
        # if epoch == num_epochs-1:
        if (epoch % save_step == 0) or (epoch == num_epochs-1):
            # エポックごとにmu, logvarを保存
            np.save(os.path.join(z_output_dir, f"mu_epoch_{epoch}.npy"), np.array(mu_list))
            np.save(os.path.join(z_output_dir, f"logvar_epoch_{epoch}.npy"), np.array(logvar_list))
            print(f"Saved latent variables for epoch {epoch} to {z_output_dir}")
        if epoch % save_step == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            # print(train_history,'train_history')
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
        
        # if epoch % save_step == 0:#追加
        #     # 各エポックでのプロットを保存
        #     train_predictions = np.concatenate(train_predictions, axis=0)
        #     print(train_predictions.shape,'train_predictions_np.concatenate')
        #     train_ground_truth = np.concatenate(train_ground_truth, axis=0)
        #     plot_predictions_and_save(train_predictions, train_ground_truth, ckpt_dir, epoch, policy_class, prefix="train")

    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:

        # 'a_hat' をスキップ
        if key == 'a_hat':
            continue

        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        # print(train_values,'train_values')
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        # plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

import matplotlib.pyplot as plt
import os

def plot_predictions_and_save(predictions, ground_truth, ckpt_dir, epoch, policy_class, prefix="train"):
    """
    各次元ごとの予測値と正解値をプロットして保存する。

    :param predictions: モデルの予測アクション（形状: [N, D]）
    :param ground_truth: 正解アクション（形状: [N, D]）
    :param ckpt_dir: 画像を保存するディレクトリ
    :param epoch: 現在のエポック数
    :param policy_class: 使用しているポリシークラス（"ACT" または "CNNMLP"）
    :param prefix: ファイル名の接頭辞（例: "train" や "val"）
    """
    num_dimensions = predictions.shape[2]  # アクションの次元数
    step_dim = 1
    os.makedirs(ckpt_dir, exist_ok=True)
    predictions_values = []
    ground_truth_values = []

    for dim in range(num_dimensions):
        plt.figure()
        print(predictions.shape,'predictions.shape')
        predictions_values.append(predictions[:,step_dim,dim])
        print(predictions_values)
        ground_truth_values.append(ground_truth[:,step_dim,dim])
        # plt.plot(predictions[:, 1,dim], label="Predicted", linestyle='-', marker='o', alpha=0.7)
        # plt.plot(np.linspace(0, epoch-1, len(predictions_values)), predictions_values[:,1,1,dim] ,label="Predicted", linestyle='-', marker='o', alpha=0.7)
        # # plt.plot(ground_truth[:, 1,dim], label="Ground Truth", linestyle='--', alpha=0.7)
        # plt.plot(np.linspace(0, epoch-1, len(ground_truth_values)), ground_truth_values[:,1,1,dim] ,label="Ground Truth", linestyle='--', alpha=0.7)
        plt.title(f"{policy_class} Predictions vs Ground Truth (Dimension {dim})")
        plt.xlabel("Sample Index")
        plt.ylabel("Action Value")
        plt.legend()
        plt.tight_layout()

        # 保存ファイル名を生成
        plot_path = os.path.join(ckpt_dir, f"{prefix}_predictions_dim_{dim}_epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
    
    
    print(f"Saved {prefix} prediction plots to {ckpt_dir}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--action_seed', action='store', type=int, help='Action seed for specific configuration', required=False)
    parser.add_argument('--action_save_dir_num' ,action='store', type=int, required=False)

    
    main(vars(parser.parse_args()))
