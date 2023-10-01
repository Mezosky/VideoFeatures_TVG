# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint

# Add decord to read videos and not images
from decord import VideoReader
from decord import cpu, gpu
import decord
import torchvision
import torchvision.transforms as T
from torchvision import transforms

# Load video metadata
from moviepy.video.io.VideoFileClip import VideoFileClip
# Set how default a torch tensor
decord.bridge.set_bridge('torch')

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoLoader(data.Dataset):
    def __init__(self, root_path, video_name, num_segments=3, n_frame=8,
                new_length=1, modality='RGB', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.video_path = root_path
        self.video_name = video_name
        self.num_segments = num_segments
        self.n_frame = n_frame
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_video()

    def _parse_video(self):
        try:
            # cambiar esto para solo un  video
            video_path = os.path.join(self.video_path, self.video_name)
            video      = VideoFileClip(video_path)
            number_fps = video.fps*video.duration
        except Exception as e:
            print(f"Failed to load video from {os.path.join(self.list_file, video)} with error {e}")
    
        self.video = VideoRecord((video_path, number_fps, 0))
        print('video length:%d' % (self.video.num_frames))

    def _get_test_indices(start_frame, end_frame, new_length, num_segments):
        tick = (end_frame - new_length + 1) / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) + start_frame for x in range(num_segments)])
        return offsets

    def __getitem__(self, index):
        if index <= self.video.num_frames // self.n_frame:
            if index != 0:    
                start_frame = (index-1)*self.n_frame
                end_frame   = (index)*self.n_frame
            else:
                start_frame = index
                end_frame   = (index+1)*self.n_frame
            segment_indices = range(start_frame, end_frame, 1)
            print(segment_indices)
        else:
            raise Exception("Index entered exceeds video length")

        return self.get(self.video, segment_indices)

    def _frames_to_pil(self, frames):
        
        t = transforms.ToPILImage()
        images_list = list()
        for i in range(frames.shape[0]):
            frame2pil = frames[i].permute(2, 0, 1)
            frame2pil = t(frame2pil)
            
            if self.modality == 'RGB' or self.modality == 'RGBDiff':
                frame2pil = frame2pil.convert('RGB')
                images_list.extend([frame2pil])
            elif self.modality == 'Flow':
                frame2pil = frame2pil.convert('RGB')
                flow_x, flow_y, _ = frame2pil.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')
                images_list.extend([x_img, y_img])
        
        return images_list

    def get(self, record, indices, new_length=1):
        
        # Get the frames
        vr = VideoReader(record._data[0], ctx=cpu(0))
        images = vr.get_batch(indices)
        images = self._frames_to_pil(images)
        process_data = self.transform(images)
        type(process_data)
        return process_data

    def __len__(self):
        return self.video.num_frames // self.n_frame


for i in range(64, 1024, 64): 
    _get_test_indices(self, record)