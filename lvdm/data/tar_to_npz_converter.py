"""
Convert fractal20220817_data from tar+pickle format (WebDataset style)
to the npz format expected by Vid2World's RT-1 dataloader.

Each tar contains sample_*.data.pickle files, one per episode.
Output npz files have the same structure as oxe_data_converter.py produces:
  - image: [T, H, W, C]  uint8
  - action: [T, 13]      float32

Usage:
    python lvdm/data/tar_to_npz_converter.py \
        --input_path OpenX-Embodiment/fractal20220817_data \
        --output_path /data/wenkai_huang/fractal20220817_data_npz \
        --num_workers 4
"""

import os
import io
import tarfile
import pickle
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def decode_jpeg(jpeg_bytes):
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img[:, :, ::-1]  # BGR -> RGB


def process_episode(episode_data):
    """Convert one episode dict to (frames, actions) arrays."""
    steps = episode_data['steps']

    frames = []
    actions = []
    for step in steps:
        obs = step['observation']
        img_bytes = obs['image']
        img = decode_jpeg(img_bytes)
        frames.append(img)

        a = step['action']
        action_vec = np.concatenate([
            a['base_displacement_vector'],           # (2,)
            a['base_displacement_vertical_rotation'], # (1,)
            a['gripper_closedness_action'],           # (1,)
            a['rotation_delta'],                      # (3,)
            a['terminate_episode'].astype(np.float32),# (3,)
            a['world_vector'],                        # (3,)
        ])  # total: 13-dim
        actions.append(action_vec)

    frames = np.array(frames, dtype=np.uint8)    # [T, H, W, C]
    actions = np.array(actions, dtype=np.float32) # [T, 13]
    return frames, actions


def process_tar(args):
    tar_path, output_path, start_idx = args
    tar_name = os.path.basename(tar_path).replace('.tar', '')
    saved = 0
    errors = 0

    with tarfile.open(tar_path, 'r') as tf:
        members = [m for m in tf.getmembers() if m.name.endswith('.data.pickle')]
        for local_i, member in enumerate(members):
            global_i = start_idx + local_i
            out_file = os.path.join(output_path, f'train_eps_{global_i:08d}.npz')

            if os.path.exists(out_file):
                saved += 1
                continue

            try:
                f = tf.extractfile(member)
                episode_data = pickle.load(f)
                frames, actions = process_episode(episode_data)
                np.savez_compressed(out_file, image=frames, action=actions)
                saved += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] {tar_name}/{member.name}: {e}")

    return saved, errors


def count_episodes_per_tar(tar_paths):
    """Count number of pickle files in each tar to compute global indices."""
    counts = []
    for tar_path in tqdm(tar_paths, desc="Counting episodes"):
        with tarfile.open(tar_path, 'r') as tf:
            n = sum(1 for m in tf.getmembers() if m.name.endswith('.data.pickle'))
        counts.append(n)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default='OpenX-Embodiment/fractal20220817_data',
                        help='Directory containing .tar files')
    parser.add_argument('--output_path', type=str,
                        default='/data/wenkai_huang/fractal20220817_data_npz',
                        help='Output directory for .npz files')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    tar_paths = sorted([
        os.path.join(args.input_path, f)
        for f in os.listdir(args.input_path)
        if f.endswith('.tar')
    ])
    print(f"Found {len(tar_paths)} tar files in {args.input_path}")

    counts = count_episodes_per_tar(tar_paths)
    print(f"Total episodes: {sum(counts)}")

    start_indices = [0] + list(np.cumsum(counts[:-1]))

    # Build task list
    tasks = [
        (tar_path, args.output_path, start_idx)
        for tar_path, start_idx in zip(tar_paths, start_indices)
    ]

    total_saved = 0
    total_errors = 0

    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="Converting tars"):
            saved, errors = process_tar(task)
            total_saved += saved
            total_errors += errors
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_tar, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting tars"):
                saved, errors = fut.result()
                total_saved += saved
                total_errors += errors

    print(f"\nDone! Saved: {total_saved} episodes, Errors: {total_errors}")
    print(f"Output: {args.output_path}")


if __name__ == '__main__':
    main()
