# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import scipy
import pickle
import random
import os
import numpy as np
import cv2
import glob
import re

from scipy import spatial
from matplotlib.pyplot import imread, figure
from matplotlib import pyplot as plt

from tqdm import tqdm
from typing import Optional, Callable, List
# -

raw_data_path = os.path.join("../../", os.environ["RAW_DATASET_PATH"])
complete_data_path = os.path.join("../../",  os.environ["COMPLETE_DATA_PATH"])
audio_data_path = os.path.join("../../", os.environ["AUDIO_DATA_PATH"])
img_data_path = os.path.join("../../", os.environ["IMG_DATA_PATH"])

# ## Explore images

files = glob.glob(f"{img_data_path}/*/*.png")

# +
figure(figsize=(8, 6), dpi=80)
img = cv2.imread(files[0],0)

# Initiate STAR detector
orb = cv2.KAZE_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,img,color=(255,0,0), flags=0)
plt.imshow(img2),plt.show()
# -

files[0].split("/")[-1]


# +
def extract_features(img_path: str,
                     algo_creator: Callable,
                     vector_size:Optional[int]=32):
    """
    Function to extract features using open cv
    objects
    
    Paramters
    ---------
    img: str
        Path of the image to be processed
    algo_creator: Callable
        open cv feature extractor creator method
    vector_size: Optional[int]
        Number indicating the size of the vector for
        feature extraction
        
    Returns
    -------
    dsc: np.ndarray
        vector containing extracted features for image
    """
    img = imread(img_path)
    alg = algo_creator()

    kps = alg.detect(img)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)
    
    if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        
    return dsc

def batch_extractor(files: List[str],
                    algo_creator: Callable,
                    algo_name: str,
                    pickle_name: str) -> str:
    """
    Function to perform feature extraction for a collection of files
    
    Parameters
    ----------
    files: List[str]
        Collection of paths containing images to be
        processed
    algo_creator: Callable
        open cv feature extractor creator method
    algo_name: str
        name indicating the used algorithm for 
        persistence purposes
    
    pickled_db_path: str
        Name for the stored file
        containing extracted features
        
    Returns
    -------
    file_name: str
        path for the persisted data file
    """
    result = {}
    for f in files:
        print('Extracting features for image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f, algo_creator)
        
    file_name = f"{pickle_name}-{algo_name}.pkl"
    with open(file_name, "wb") as fp:
        pickle.dump(result, fp)
        
    return file_name


# -

tqdm(batch_extractor(files, cv2.KAZE_create, 'KAZE', 'features'))

# +
# tqdm(batch_extractor(files, cv2.ORB_create, 'ORB', 'features'))

# +
# tqdm(batch_extractor(files, cv2.SIFT_create, 'SIFT', 'features'))
# -

tqdm(batch_extractor(files, cv2.AKAZE_create, 'AKAZE', 'features'))


class Matcher:
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as fp:
            self.data = pickle.load(fp)
            
        self.names = []
        self.matrix = []
        
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)
        
    def cos_cdist(self, vector):
        """
        Compute cosine distance between the image (vector) and
        the db of images

        Parameters
        ---------
        vector: np.ndarray

        """
        v = vector.reshape(1, -1)
        return spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)
        
    def match(self, img_path, algo_creator, top_n=5):
        """
        Parameters
        ----------
        """
        features = extract_features(img_path, algo_creator)
        # Obtiene distancias
        img_distances = self.cos_cdist(features)
        # Get the idx of the most similar images
        nearest_ids = np.argsort(img_distances)[:top_n].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


ma = Matcher("features-KAZE.pck")

samples = random.sample(files, 3)

for s in samples:
    show_img(s)


# +
def show_img(path):
    img = imread(path)
    plt.imshow(img)
    plt.show()

def run(files, data_path, file_format,
        algo_creator, algo_name,
        pickle_name,
        n_sample=5, top_n=3,
        pickle_path=None
       ):
    if pickle_path is None:
        files = glob.glob(f"{data_path}/*/*.{file_format}")
        pickle_path = batch_extractor(files, algo_creator,
                                  algo_name, pickle_name)
    
    sample = random.sample(files, n_sample)
    
    ma = Matcher(pickle_path)
    
    for s in sample:
        print('Query image ==========================================')
        print(s)
        show_img(s)
        names, match = ma.match(s, algo_creator, top_n=top_n)
        print('Result images ========================================')
        for i in range(top_n):
            genre_dir = re.findall("[a-zA-Z]+", names[i])[0]
            print(names[i])
            show_img(os.path.join(data_path, genre_dir, names[i]))


# -

run(files, img_data_path, 'png', cv2.AKAZE_create, 'AKAZE', 'features', pickle_path='features-KAZE.pck')
