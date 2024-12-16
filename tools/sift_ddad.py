from __future__ import absolute_import, division, print_function
import os
import cv2 as cv
import pickle
from tqdm import tqdm
import multiprocessing
import time
from functools import wraps
import copyreg

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


# Decorator function to measure the time taken by a function
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        wrapper.times.append(elapsed_time)
        print(f"Time taken for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    wrapper.times = []
    return wrapper

@timeit
def detect_and_compute(img, sift):
    return sift.detectAndCompute(img, None)

@timeit
def load_processed_frames(root_path, camera_names):
    processed_frames = set()
    for scene_name in os.listdir(root_path):
        for camera_name in camera_names:
            camera_path = os.path.join(root_path, scene_name, 'sift', camera_name)
            if os.path.exists(camera_path):
                for file_name in os.listdir(camera_path):
                    frame_id = os.path.splitext(file_name)[0]
                    processed_frames.add(frame_id)
    return processed_frames

def process(args):
    frame_id, info, root_path, rgb_path, camera_names, processed_frames = args
    sift = cv.xfeatures2d.SIFT_create(edgeThreshold=8, contrastThreshold=0.01)
    scene_name = info[frame_id]['scene_name']
    for camera_name in camera_names:
        to_save = {}
        os.makedirs(os.path.join(root_path, scene_name, 'sift', camera_name), exist_ok=True)
        save_path = os.path.join(root_path, scene_name, 'sift', camera_name, frame_id + '.pkl')
        if frame_id in processed_frames:
            continue

        inputs = cv.imread(os.path.join(rgb_path, scene_name, 'rgb', camera_name, frame_id + '.png'))
        img1 = cv.cvtColor(inputs, cv.COLOR_RGB2GRAY)
        kp1, des1 = detect_and_compute(img1, sift)
        to_save['kp'] = kp1
        to_save['des'] = des1

        with open(save_path, 'wb') as f:
            pickle.dump(to_save, f)

def main():
    root_path = '../data/ddad/sift'
    rgb_path = '../data/ddad/raw_data'
    camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']

    with open('../datasets/ddad/info_train.pkl', 'rb') as f:
        info = pickle.load(f)

    # Load already processed frames and measure the time
    processed_frames = load_processed_frames(root_path, camera_names)

    # Filter out already processed frames
    info_list = [frame_id for frame_id in info.keys() if frame_id not in processed_frames]

    # Calculate total and average time for loading processed frames
    total_load_time = sum(load_processed_frames.times)
    average_load_time = total_load_time / len(load_processed_frames.times)
    print(f"Total time for loading processed frames: {total_load_time:.4f} seconds")
    print(f"Average time per frame for loading processed frames: {average_load_time:.4f} seconds")

    # Process remaining frames
    p = multiprocessing.Pool(8)
    args = [(frame_id, info, root_path, rgb_path, camera_names, processed_frames) for frame_id in info_list]
    for _ in tqdm(p.imap_unordered(process, args), total=len(info_list)):
        pass
    p.close()
    p.join()

    # Calculate total and average time for sift.detectAndCompute
    total_time = sum(detect_and_compute.times)
    average_time = total_time / len(detect_and_compute.times)
    print(f"Total time for sift.detectAndCompute: {total_time:.4f} seconds")
    print(f"Average time per frame for sift.detectAndCompute: {average_time:.4f} seconds")

if __name__ == "__main__":
    main()