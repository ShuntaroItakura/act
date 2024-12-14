# import rospy
# from geometry_msgs.msg import PoseStamped, Twist
# from sensor_msgs.msg import JointState
# from std_srvs.srv import Empty
# import time

# def make_gazebo_env(task_name):
#     # rospy.init_node("gazebo_env_control", anonymous=True)
    
#     arm_publisher = rospy.Publisher('/hsrb/arm_controller/command', JointState, queue_size=10)
#     gripper_publisher = rospy.Publisher('/hsrb/gripper_controller/command', JointState, queue_size=10)
#     base_velocity_publisher = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
    
#     reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
#     reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

#     return {
#         "arm_publisher": arm_publisher,
#         "gripper_publisher": gripper_publisher,
#         "base_velocity_publisher": base_velocity_publisher,
#         "reset_world": reset_world,
#         "reset_simulation": reset_simulation
#     }

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
import time

import math
import moveit_commander
import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import numpy as np
# ROSノードの初期化
# rospy.init_node("arm_control_node")


# MoveItの初期化
# moveit_commander.roscpp_initialize(["joint_states:=/hsrb/joint_states"])
whole_body = moveit_commander.MoveGroupCommander("whole_body")


def make_gazebo_env(task_name, rgbd):
    """
    シミュレーション環境を初期化します。この関数は環境をリセットするための簡易インターフェースを提供します。
    """
    rospy.loginfo(f"Gazebo environment initialized for task: {task_name}")
    return GazeboHSRTask(rgbd)  # GazeboHSRTaskにrgbdを渡す



def get_action(target_joint_angles):
    """
    指定された目標角度に基づいてHSRの関節を移動させ、現在の状態を返す。
    
    Parameters:
    - target_joint_angles (list of float): 9つの目標関節角度のリスト
    
    Returns:
    - current_joint_values (list of float): 現在の関節角度
    """
    # 目標関節角度を設定
    whole_body.set_joint_value_target(target_joint_angles)
    
    # 目標角度への移動を実行
    success = whole_body.go()
    if not success:
        rospy.logwarn("Failed to move to the target joint angles.")
    
    # 現在の関節角度を取得
    current_joint_values = whole_body.get_current_joint_values()
    return current_joint_values


class Observation:
    def __init__(self, observation_dict):
        self.observation = observation_dict
class GazeboHSRTask:
    def __init__(self, rgbd):
        self.rgbd = rgbd  # カメラから画像を取得するオブジェクト
        

    def reset(self):
        """
        環境をリセットし、初期の観測データをObservation形式で返します。
        """
        rospy.loginfo("Environment reset.")
        time.sleep(1)
        initial_observation = self.get_observation()
        return Observation(initial_observation)
    
    def step(self, target_qpos):
        """
        指定された関節角度に基づいてHSRの関節を移動させ、観測データをObservation形式で返します。
        
        Parameters:
        - target_qpos (list of float): 目標関節角度のリスト
        
        Returns:
        - Observationオブジェクト: 現在の観測データ
        - float: ダミーの報酬値
        - bool: ダミーのエピソード終了フラグ
        """
        # numpy.float32からfloat64への変換
        target_qpos = np.array(target_qpos, dtype=np.float64)  # float64にキャスト

        # # 関節の範囲にクリップ
        # # 関節名とその制限に基づいて、各目標角度をクリップ
        # joint_names = whole_body.get_active_joints()
        # for i, joint_name in enumerate(joint_names):
        #     min_limit, max_limit = self.joint_limits.get(joint_name, (-np.inf, np.inf))
        #     target_qpos[i] = np.clip(target_qpos[i], min_limit, max_limit)
        print(whole_body.get_active_joints())
        target_qpos = np.append(target_qpos, -2.2631540108619674e-06)

        if target_qpos[3] < 0:
            target_qpos[3] = 0

        Joint: arm_lift_joint, Min: 0.0, Max: 0.69
Joint: arm_flex_joint, Min: -2.62, Max: 0.0
Joint: arm_roll_joint, Min: -2.09, Max: 3.84
Joint: wrist_flex_joint, Min: -1.92, Max: 1.22
Joint: wrist_roll_joint, Min: -1.92, Max: 3.67
        # target_qpos[4]= -target_qpos[4]
        # target_qpos[6] = -1.6
        # target_qpos =[0.13780725155314386, 0.41341716295430186, 0.015719038596431728, 0.5532596330349033, -0.2609849722632612, -0.0403722346803955, -0.18901872789763327, 0.031810622303098765, 0.0]
        
        print(target_qpos,'target_qpos')
        
        # whole_body.set_planning_time(5.0)  # 計画時間を10秒に設定
        whole_body.set_joint_value_target(target_qpos)
        
        # ゴール許容値や計画時間の設定（必要に応じて変更）
        # whole_body.set_goal_tolerance(0.01)  # 許容誤差を設定
        

        # 動作実行
        success = whole_body.go()

        if not success:
            rospy.logwarn(f"Failed to move to the target joint angles: {target_qpos}")
            rospy.logwarn(f"Current observation: {self.get_observation()}")  # 現在の状態を確認

        return Observation(self.get_observation()), 0, False

    def get_observation(self):
        """
        ロボットの観測データ（位置や画像など）を取得します。
        """
        # 画像データを取得し、"hand_camera" キーに設定
        hand_camera_image = self.rgbd.get_image()
        observation_data = {
            "qpos": whole_body.get_current_joint_values(),
            "images": {"hand_camera": hand_camera_image}
        }
        print(f"Current joint values: {whole_body.get_current_joint_values()}")
        print(f"Camera image checksum: {np.sum(hand_camera_image)}")  # 画像内容が異なるかを確認
        return observation_data
    
    
    def compute_reward(self, observation):
        """
        観測から報酬を計算します（任意の条件に基づく）。
        """
        reward = 0
        # 例: 目標状態に基づいて報酬を計算
        return reward

    def check_done(self, observation):
        """
        エピソードの終了条件を確認します。
        """
        done = False
        # 例: エピソードが終了する条件
        return done
# 
def test_simulation(task_name):
    env = make_gazebo_env(task_name)
    gazebo_task = GazeboHSRTask(env)
    num_rollouts = 5  # テストのためのエピソード数
    max_timesteps = 100  # 各エピソードの最大ステップ数

    for rollout_id in range(num_rollouts):
        print(f"Rollout {rollout_id+1}")
        gazebo_task.reset_environment()
        episode_reward = 0

        for t in range(max_timesteps):
            action = [0] * 8  # アクションのダミーデータ
            observation, reward, done = gazebo_task.step(action)
            episode_reward += reward
            print(f"Step {t+1}, Reward: {reward}, Done: {done}")

            if done:
                break

        print(f"Total reward for rollout {rollout_id+1}: {episode_reward}")

if __name__ == "__main__":
    test_simulation("sim_hsr_pick")
