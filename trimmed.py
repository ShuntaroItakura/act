import os
import h5py
import numpy as np

def trim_and_remove_last_episodes(input_directory, output_directory, episode_len=400, threshold_factor=100, remove_last_episodes=1000):
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)

    for filename in sorted(os.listdir(input_directory)):  # ファイルをソートして逐次処理
        if filename.endswith(".hdf5"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            print(f"Processing file: {input_file_path}")

            with h5py.File(input_file_path, 'r') as h5_file_in:
                try:
                    with h5py.File(output_file_path, 'w') as h5_file_out:
                        # 必要ならチャンク化を適用
                        qpos_dataset = ensure_chunked_dataset(h5_file_in, 'observations/qpos', h5_file_out)
                        images_dataset = ensure_chunked_dataset(h5_file_in, 'observations/images/hand_camera', h5_file_out)
                        action_dataset = ensure_chunked_dataset(h5_file_in, 'action', h5_file_out)

                        joint_values = qpos_dataset[:, 4]

                        # エピソードのトリミング範囲を決定
                        total_length = len(joint_values)
                        trim_end_index = max(0, total_length - remove_last_episodes)
                        print(f"Removing last {remove_last_episodes} episodes. Keeping first {trim_end_index} steps.")

                        trimmed_indices = list(range(trim_end_index))

                        # トリム後のデータを取得
                        trimmed_images = images_dataset[trimmed_indices]
                        trimmed_qpos = qpos_dataset[trimmed_indices]
                        trimmed_action = action_dataset[trimmed_indices]

                        # トリム後の長さが不足する場合、最後の状態を補完
                        remaining_length = len(trimmed_indices)
                        if remaining_length < episode_len:
                            padding_steps = episode_len - remaining_length
                            print(f"Extending the dataset with {padding_steps} steps using the last state.")

                            # 最後の状態を取得してパディング
                            last_image = trimmed_images[-1]
                            last_qpos = trimmed_qpos[-1]
                            last_action = trimmed_action[-1]

                            padded_images = np.repeat(last_image[np.newaxis, ...], padding_steps, axis=0)
                            padded_qpos = np.repeat(last_qpos[np.newaxis, ...], padding_steps, axis=0)
                            padded_action = np.repeat(last_action[np.newaxis, ...], padding_steps, axis=0)

                            # トリム後のデータにパディングを追加
                            new_images = np.vstack([trimmed_images, padded_images])
                            new_qpos = np.vstack([trimmed_qpos, padded_qpos])
                            new_action = np.vstack([trimmed_action, padded_action])

                            # 出力データセットに保存（逐次保存）
                            save_dataset(h5_file_out, 'observations/images/hand_camera', new_images)
                            save_dataset(h5_file_out, 'observations/qpos', new_qpos)
                            save_dataset(h5_file_out, 'action', new_action)
                        else:
                            # エピソード長を満たしている場合はトリムのみ（逐次保存）
                            save_dataset(h5_file_out, 'observations/images/hand_camera', trimmed_images[:episode_len])
                            save_dataset(h5_file_out, 'observations/qpos', trimmed_qpos[:episode_len])
                            save_dataset(h5_file_out, 'action', trimmed_action[:episode_len])

                        print(f"File {output_file_path} successfully processed.")

                except Exception as e:
                    print(f"Error processing file {input_file_path}: {e}")

def ensure_chunked_dataset(h5_file_in, dataset_path, h5_file_out, chunk_size=1000):
    """
    Utility function to ensure datasets are chunked for efficient slicing.
    """
    if dataset_path not in h5_file_in:
        raise KeyError(f"Dataset {dataset_path} not found in HDF5 file.")

    dataset_in = h5_file_in[dataset_path]
    if dataset_path in h5_file_out:
        del h5_file_out[dataset_path]  # 既存のデータセットを削除

    dataset_out = h5_file_out.create_dataset(
        dataset_path,
        shape=dataset_in.shape,
        dtype=dataset_in.dtype,
        chunks=True,
        compression="gzip",
    )

    # データをチャンク単位でコピー
    for i in range(0, dataset_in.shape[0], chunk_size):
        end = min(i + chunk_size, dataset_in.shape[0])
        dataset_out[i:end] = dataset_in[i:end]
        print(f"Copied chunk {i}-{end} for dataset {dataset_path}.")

    # 属性をコピー
    for attr_name, attr_value in dataset_in.attrs.items():
        dataset_out.attrs[attr_name] = attr_value

    return dataset_out

def save_dataset(h5_file_out, dataset_path, data, chunk_size=1000):
    """
    Save dataset to the output HDF5 file incrementally to reduce memory usage.
    """
    if dataset_path in h5_file_out:
        del h5_file_out[dataset_path]  # 既存のデータセットを削除
    dataset_out = h5_file_out.create_dataset(
        dataset_path,
        shape=data.shape,
        dtype=data.dtype,
        chunks=True,
        compression="gzip",
    )
    for i in range(0, data.shape[0], chunk_size):
        end = min(i + chunk_size, data.shape[0])
        dataset_out[i:end] = data[i:end]
        print(f"Saved chunk {i}-{end} for dataset {dataset_path}.")

# 実行例
trim_and_remove_last_episodes(
    input_directory="/home/developer/workspace/act/dataset_dir/sim_hsr_pick1124/",
    output_directory="/home/developer/workspace/act/dataset_dir/sim_hsr_pick_trimmed/",
    remove_last_episodes=1000
)
