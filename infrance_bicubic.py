import cv2
import numpy as np
import os

def upscale_video_bicubic(input_video_path, output_video_path):
    """
    Upscales a video 4x from an assumed 214x120 base using bicubic interpolation.

    Args:
        input_video_path (str): Path to the input MP4 video.
        output_video_path (str): Path to save the upscaled MP4 video.
    """
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_width = 214
    base_height = 120
    scale_factor = 4

    target_width = base_width * scale_factor
    target_height = base_height * scale_factor

    print(f"Input video: {input_video_path}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Assuming frames will be treated as {base_width}x{base_height} for 4x upscaling.")
    print(f"Target upscaled resolution: {target_width}x{target_height}")

    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (target_width, target_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames if total_frames > 0 else 'Unknown (streaming?)'}")

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        
        
        
        
        

        
        upscaled_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        
        out.write(upscaled_frame)

        frame_count += 1
        if frame_count % 100 == 0: 
            if total_frames > 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
            else:
                print(f"Processed {frame_count} frames...")


    
    cap.release()
    out.release()
    cv2.destroyAllWindows() 

    print(f"\nSuccessfully upscaled video saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}")

def create_dummy_video(filename="dummy_input_214x120.mp4", width=214, height=120, fps=30, duration_sec=3):
    """Creates a dummy video for testing purposes."""
    if os.path.exists(filename):
        print(f"Dummy video '{filename}' already exists. Skipping creation.")
        return

    print(f"Creating dummy video '{filename}' ({width}x{height} @ {fps}fps, {duration_sec}s)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))
    if not out.isOpened():
        print("Error: Could not create dummy video writer.")
        return

    for i in range(fps * duration_sec):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        color_val = (i * 5) % 256
        frame[:, :, i % 3] = color_val  
        cv2.putText(frame, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(frame)
    out.release()
    print(f"Dummy video '{filename}' created successfully.")


if __name__ == "__main__":
    
    input_video_name = "media\lr_output_video.mp4" 
    output_video_name = "my_video_upscaled_bicubic.mp4"

    
    
    
    
    
    if not os.path.exists(input_video_name):
        print(f"Input video '{input_video_name}' not found.")
        create_dummy_video(filename=input_video_name, width=214, height=120, fps=25, duration_sec=5)
        
        


    
    if os.path.exists(input_video_name):
        upscale_video_bicubic(input_video_name, output_video_name)
    else:
        print(f"Cannot proceed: Input video '{input_video_name}' still not found after attempting to create dummy.")