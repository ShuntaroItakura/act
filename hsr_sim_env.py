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
gripper = moveit_commander.MoveGroupCommander("gripper")


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
    
    gripper.set_joint_value_target('hand_motor_joint',0.0)
    success = gripper.go()

    
    if not success:
        rospy.logwarn("Failed to move to the target gripper angles.")
    
    # 現在の関節角度を取得
    current_joint_values = whole_body.get_current_joint_values()
    current_gripper_values = gripper.get_current_joint_values()

    return current_joint_values[:8]+[current_joint_values[2]]


class Observation:
    def __init__(self, observation_dict):
        self.observation = observation_dict
        # print(observation_dict,'observation_dict')
class GazeboHSRTask:
    def __init__(self, rgbd):
        self.rgbd = rgbd  # カメラから画像を取得するオブジェクト
        self.joint_limits = {
            'arm_lift_joint': (0.0, 0.69),
            'arm_flex_joint': (-2.62, 0.0),
            'arm_roll_joint': (-2.09, 3.84),
            'wrist_flex_joint': (-1.92, 1.22),
            'wrist_roll_joint': (-1.92, 3.67),
            'hand_motor_joint': (0.0,1.0)
        }

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

        # 関節の範囲にクリップ
        joint_names = whole_body.get_active_joints()
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint_name]
                target_qpos[i] = np.clip(target_qpos[i], min_limit, max_limit)

        target_joint_qpos = np.append(target_qpos[:8], -2.2631540108619674e-06)

        print(target_joint_qpos, 'target_joint_qpos after clipping')
        

        whole_body.set_joint_value_target(target_joint_qpos)

        # 動作実行
        success = whole_body.go()

        if not success:
            rospy.logwarn(f"Failed to move to the target joint angles: {target_qpos}")
            rospy.logwarn(f"Current observation: {self.get_observation()}")  # 現在の状態を確認
        gripper.set_joint_value_target([0,0,target_qpos[-1],0,0])
        gripper.go()

        return Observation(self.get_observation()), 0, False

    

    def get_observation(self):
        """
        ロボットの観測データ（位置や画像など）を取得します。
        """
        # 画像データを取得し、"hand_camera" キーに設定
        hand_camera_image = self.rgbd.get_image()
        # print(hand_camera_image,'hand_camera_image')
        observation_data = {
            "qpos": whole_body.get_current_joint_values()[:8]+[gripper.get_current_joint_values()[2]],
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
            action = [0] * 9  # アクションのダミーデータ
            observation, reward, done = gazebo_task.step(action)
            episode_reward += reward
            print(f"Step {t+1}, Reward: {reward}, Done: {done}")

            if done:
                break

        print(f"Total reward for rollout {rollout_id+1}: {episode_reward}")

if __name__ == "__main__":
    test_simulation("sim_hsr_pick")
