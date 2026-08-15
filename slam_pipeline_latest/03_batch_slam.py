"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
import hashlib
from tqdm import tqdm
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask


# %%
def get_docker_safe_name(video_dir):
    safe_name = ''.join(
        c if c.isalnum() else '_' for c in video_dir.name)
    safe_name = safe_name.strip('_')[:48] or 'video'
    digest = hashlib.sha1(str(video_dir).encode('utf-8')).hexdigest()[:10]
    return f"umi_slam_{os.getpid()}_{safe_name}_{digest}"


def runner(cmd, cwd, stdout_path, stderr_path, timeout,
        cidfile_path=None, container_name=None, **kwargs):
    cidfile_path = pathlib.Path(cidfile_path) if cidfile_path is not None else None
    if cidfile_path is not None and cidfile_path.exists():
        cidfile_path.unlink()

    with stdout_path.open('w') as stdout_file, stderr_path.open('w') as stderr_file:
        proc = subprocess.Popen(cmd,
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            **kwargs)
        try:
            returncode = proc.wait(timeout=timeout)
            return subprocess.CompletedProcess(cmd, returncode)
        except subprocess.TimeoutExpired as e:
            cid = None
            if cidfile_path is not None and cidfile_path.exists():
                cid = cidfile_path.read_text().strip()

            target = cid or container_name
            stderr_file.write(
                f"\n[03_batch_slam] Timeout after {timeout:.1f}s.\n")
            if target:
                stderr_file.write(
                    f"[03_batch_slam] Stopping docker container {target}.\n")
                subprocess.run(
                    ['docker', 'stop', '--time', '10', target],
                    stdout=stderr_file,
                    stderr=stderr_file)
                subprocess.run(
                    ['docker', 'rm', '-f', target],
                    stdout=stderr_file,
                    stderr=stderr_file)

            if proc.poll() is None:
                proc.kill()
                proc.wait()
            return e
        finally:
            if cidfile_path is not None and cidfile_path.exists():
                cidfile_path.unlink()


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nfm', '--no_finger_mask', is_flag=True, default=False, help="Do not mask bottom finger band (use if ArUco/QR tags sit there).")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="No SLAM mask at all (mirror+finger off); ORB-SLAM sees full frame.")
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull, no_finger_mask, no_mask):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                if not no_mask:
                    slam_mask = np.zeros((3000, 4000), dtype=np.uint8)
                    slam_mask = draw_predefined_mask(
                        slam_mask, color=255, mirror=True, gripper=False,
                        finger=not no_finger_mask)
                    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
                container_name = get_docker_safe_name(video_dir)
                cidfile_path = video_dir.joinpath('slam_container.cid')

                # run SLAM
                cmd = [
                    'docker',
                    'run',
                    '--rm', # delete after finish
                    '--name', container_name,
                    '--cidfile', str(cidfile_path),
                    '--label', 'umi.slam.stage=03_batch_slam',
                    '--label', f'umi.slam.input_dir={input_dir}',
                    '--label', f'umi.slam.video_dir={video_dir}',
                    '--volume', str(video_dir) + ':' + '/data',
                    '--volume', '/home/oem/dkim/indy_umi/config' + ':' + '/config',
                    '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
                    docker_image,
                    '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
                    '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
                    # '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
                    '--setting', '/config/gopro13_ultrawide_2000x1500.yaml',
                    '--input_video', str(video_path),
                    '--input_imu_json', str(json_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_mount_target),
                    '--max_lost_frames', str(max_lost_frames)
                ]
                if not no_mask:
                    cmd.extend(['--mask_img', str(mask_path)])

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout,
                    cidfile_path=cidfile_path, container_name=container_name))
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done! Result:")
    print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
