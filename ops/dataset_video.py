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

# debugger
import ipdb

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


class TSNDataSetVideos(data.Dataset):
    def __init__(self, list_file, num_segments=3, 
                new_length=1, modality='RGB', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.list_file = list_file
        self.num_segments = num_segments
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

        self._parse_list_video()

    def _parse_list_video(self):
        list_videos = os.listdir(self.list_file)
        tmp = []
        for video in list_videos:
            try:
                video_path = os.path.join(self.list_file, video)
                video      = VideoFileClip(video_path)
                number_fps = video.fps*video.duration
                tmp.append((video_path, number_fps, 0))
            except Exception as e:
                print(f"Failed to load video from {os.path.join(self.list_file, video)} with error {e}")
        
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        print('video number:%d' % (len(self.video_list)))

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._get_test_indices(record)
        frames_output = self.get(record, segment_indices)
        return frames_output

    def _frames_to_pil_video(self, frames):
        
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
        
        t = transforms.ToPILImage()
        idx_image = []
        for seg_ind in indices:
            p = int(seg_ind)
            idx_image.append(p)
            for i in range(new_length):
                if p < record.num_frames:
                    p += 1
                    idx_image.append(p)
        idx_image.sort()
        
        # Get the frames
        vr = VideoReader(record._data[0], ctx=cpu(0))
        images = vr.get_batch(idx_image)
        images = self._frames_to_pil_video(images)
        
        process_data = self.transform(images)
        print(process_data.shape)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)