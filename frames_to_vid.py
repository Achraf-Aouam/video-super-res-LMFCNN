import cv2
import os

def create_video_from_frames(frame_dir, output_file, fps=30):
    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frames:
        raise ValueError("No frames found in directory")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_file}")

# Create video from HR frames
create_video_from_frames("inference_results/lr"  , "lr_output_video.mp4", fps=12)