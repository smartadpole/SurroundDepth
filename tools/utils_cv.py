import cv2 as cv
import copyreg

__all__ = ['_pickle_keypoints']

# Function to pickle OpenCV keypoints
def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)

# Register the keypoint pickling function
copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)