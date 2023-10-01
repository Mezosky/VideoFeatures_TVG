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
        self.loaded_frames = None

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
            print(f"The video has {number_fps} fps")
        except Exception as e:
            print(f"Failed to load video from {os.path.join(self.list_file, video)} with error {e}")
    
        vr = VideoReader(video_path, ctx=cpu(0))
        self.indices_list = self._get_all_indices_list(len(vr), self.n_frame, self.num_segments)
        self.cache_list   = self._cache_batch(self.indices_list)
        print(f'video length: {len(self.indices_list)}')
        

    def _get_test_indices(self, start_frame, end_frame, new_length, num_segments):
        tick = (end_frame - new_length + 1) / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) + start_frame for x in range(num_segments)])
        return offsets


    def _get_all_indices_list(self, total_frames, num_frames, num_segments):

        idx_list = list()
        list_frames = range(0, total_frames, num_frames*num_frames)
        sum_indices = self._get_test_indices(0, num_frames*num_frames, 1, num_segments)
        
        for idx, frames in enumerate(list_frames): 
            
            if idx == 0:
                sampled_frames = sum_indices + frames
                idx_list.append(sampled_frames)
            else:
                sampled_frames = sum_indices + frames
                if sum(total_frames < sampled_frames) > 0:
                    pass
                else:
                    idx_list.append(sampled_frames)
        
        if idx_list[-1][-1] == total_frames:
            idx_list[-1][-1] = idx_list[-1][-1] - 1
            
        return idx_list

    def _cache_batch(self, frames_list):
    
        count = 0
        i = 0
        output_list = list()
        aux_list = list()
        idx_frame_list = list()
        for idx, batch in enumerate(frames_list):
            
            count += len(batch)
            aux_list.extend(batch)
            idx_frame_list.append(idx)
            
            if count == 400:
                output_list.append(
                    (i,
                    np.array(aux_list), 
                    np.array(idx_frame_list)
                    )
                )
                count = 0
                aux_list = list()
                idx_frame_list = list()
                i += 1
        
        if count < 400 and len(aux_list) != 0:
            output_list.append(
                (i, 
                np.array(aux_list), 
                np.array(idx_frame_list)
                )
            )

        return output_list

    def __getitem__(self, index):

        def search_cache(index, cache_list):
            for idx, cache in enumerate(cache_list):
                if index in cache[2]:
                    video_path = os.path.join(self.video_path, self.video_name)
                    vr         = VideoReader(video_path, ctx=cpu(0))
                    return vr.get_batch(cache[1]), cache
                
        if index > len(self.indices_list):
            raise Exception("Index entered exceeds video length")
            
        if self.loaded_frames == None:
            self.loaded_frames, self.cache_loaded = search_cache(index, self.cache_list)
        elif index not in self.cache_loaded[2]:
            self.loaded_frames, self.cache_loaded = search_cache(index, self.cache_list)
        elif index in self.cache_loaded[2]:
            pass
        else:
            raise Exception("Not implemented")
        
        frames_idxs = self.indices_list[index]
        cache_idxs  = self.cache_loaded[1]
        min_index   = int(np.where(cache_idxs == frames_idxs.min())[0])
        max_index   = int(np.where(cache_idxs == frames_idxs.max())[0])+1

        return self.get(self.loaded_frames[min_index: max_index])

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

    def get(self, images, new_length=1):
        
        # Get the frames
        images = self._frames_to_pil(images)
        images = self.transform(images)
        return images

    def __len__(self):
        return len(self.indices_list)