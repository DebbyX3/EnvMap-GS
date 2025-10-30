# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
import cv2

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, read_images_from_dir, save_video

def get_video_framerate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't open video file:", video_path)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--input_images_dir', type=str, default=None, help='Path to directory containing input images (as a sequence)')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video/images, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_npz_fps', type=int, default=-1, help='save npz depths every X fps, -1 means the original fps, so save all frames')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if args.input_images_dir is not None:
        # Read image sequence from directory
        frames, target_fps, image_names = read_images_from_dir(args.input_images_dir, args.max_len)
        video_name = os.path.basename(os.path.normpath(args.input_images_dir))
    elif args.input_video is not None:
        # Read frames from video
        frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
        image_names = None
        video_name = os.path.basename(args.input_video)
    else:
        raise ValueError('Either --input_video or --input_images_dir must be specified.')

    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    # Only save videos if input is a video, not an image sequence
    if args.input_images_dir is None:
        save_video(frames, processed_video_path, fps=fps)
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        # Determine framerate for naming and interval
        if args.input_images_dir is not None:
            framerate = target_fps if target_fps > 0 else 30  # Default to 30 if not set
        else:
            framerate = get_video_framerate(args.input_video)

        if args.save_npz_fps == -1:
            fps_extract = framerate
        else:
            fps_extract = args.save_npz_fps

        frame_interval = int(framerate / fps_extract) if fps_extract > 0 else 1

        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npz')
        os.makedirs(depth_npz_dir, exist_ok=True)

        for frame_number, depth in enumerate(depths):
            timestamp = frame_number / framerate
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            milliseconds = int((timestamp * 1000) % 1000)
            timestamp_str = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"

            # For image sequence, optionally use image filename for output
            if args.input_images_dir is not None and image_names is not None:
                base_name = os.path.splitext(os.path.basename(image_names[frame_number]))[0]
                out_name = f"{base_name}_depth.npz"
            else:
                out_name = f"{timestamp_str}_depth.npz"

            if frame_number % frame_interval == 0:
                depth_npz_path = os.path.join(depth_npz_dir, out_name)
                np.savez_compressed(depth_npz_path, depth=depth)

    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for frame_number, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/{frame_number:04d}_depth.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()