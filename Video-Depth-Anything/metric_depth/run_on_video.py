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
from utils.dc_utils import read_video_frames, save_video

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
    parser.add_argument('--input_video', type=str, default='../assets/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
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

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        # Get original video framerate
        framerate_video = get_video_framerate(args.input_video)

        if args.save_npz_fps == -1:
            fps_extract = framerate_video
        else:
            fps_extract = args.save_npz_fps

        # Calcola il passo (interval) per estrarre il numero giusto di frame al secondo
        frame_interval = int(framerate_video / fps_extract)

        depth_npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npz')
        os.makedirs(depth_npz_dir, exist_ok=True)

        for frame_number, depth in enumerate(depths):
            # Calcola il timestamp in secondi
            timestamp = frame_number / framerate_video
            
            # Calcola ore, minuti, secondi e decimali dal timestamp
            hours = int(timestamp // 3600)  # Ore
            minutes = int((timestamp % 3600) // 60)  # Minuti
            seconds = int(timestamp % 60)  # Secondi
            milliseconds = int((timestamp * 1000) % 1000)  # Millisecondi
            
            # Crea il nome del file con il formato ore_minuti_secondi_decimali
            timestamp_str = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"

            # Seleziona solo i frame da estrarre, in base agli FPS desiderati
            if frame_number % frame_interval == 0:
                depth_npz_path = os.path.join(depth_npz_dir, f'{timestamp_str}_depth.npz')
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

    


