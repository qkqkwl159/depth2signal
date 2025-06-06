import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform
import threading
import os
import datetime

import torch
import torchvision.transforms as transforms
from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

# Load the MiDaS model
# Using the small model for performance
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval() # Set model to evaluation mode

# Load transforms to resize and normalize the image
# The transforms depend on the model type
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "MiDaS_small":
    transform = transforms.small_transform
else:
    transform = transforms.dpt_transform

# Basic VideoCapturer class based on the existing code
class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """Initialize and start video capture"""
        try:
            # Use default backend first, then try specific indices
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)

            if not self.cap or not self.cap.isOpened():
                 # Fallback for some systems, try index 0
                 self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
                 if not self.cap or not self.cap.isOpened():
                     raise RuntimeError("Failed to open camera")

            # Configure format
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

# Example usage:
if __name__ == "__main__":
    camera_index = 0 # Usually 0 is the default webcam
    capturer = VideoCapturer(camera_index)

    if capturer.start():
        print(f"Webcam {camera_index} started successfully.")
        try:
            average_depth_history = [] # Initialize history for outlier detection
            
            # Create a directory named with current date and time
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"./depth_data/{current_time_str}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Data will be saved in: {output_dir}")
            
            # Define data file path and open it
            data_file_path = os.path.join(output_dir, "depth_data.csv")
            data_file = open(data_file_path, "w")
            data_file.write("Frame,AverageDepth,MinDepth,MaxDepth,IsOutlier\n") # Write header with min/max depth
            
            frame_count = 0 # Initialize frame counter

            # Define video file path and setup VideoWriter
            video_file_path = os.path.join(output_dir, "webcam_feed.avi")
            # Initialize video_writer to None, will be set after the first frame is processed
            video_writer = None
            print(f"Video will be saved to: {video_file_path}")

            while capturer.is_running:
                ret, frame = capturer.read()
                if not ret:
                    print("Failed to read frame from webcam.")
                    break

                # Initialize outlier flag for the current frame
                is_outlier = False

                # --- Depth Estimation --- #
                # Convert the frame to RGB (MiDaS expects RGB)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Apply transforms and move to device
                input_batch = transform(img).to(device)

                with torch.no_grad():
                    prediction = midas(input_batch)

                    # Resize prediction to original input size
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_map = prediction.cpu().numpy()

                # --- Outlier Detection --- #
                # Calculate average depth for the current frame
                current_average_depth = np.mean(depth_map)

                # Store recent average depths for outlier detection (using a simple list for history)
                history_size = 30 # Number of recent frames to consider
                average_depth_history.append(current_average_depth)
                if len(average_depth_history) > history_size:
                    average_depth_history.pop(0) # Remove the oldest entry

                # Check for outliers if enough data is available
                if len(average_depth_history) == history_size:
                    mean_depth = np.mean(average_depth_history)
                    std_depth = np.std(average_depth_history)
                    outlier_threshold = 3.0
                    if std_depth > 0:
                        z_score = (current_average_depth - mean_depth) / std_depth
                        if abs(z_score) > outlier_threshold:
                            print("이상치가 감지되었습니다!")
                            is_outlier = True

                # --- End Outlier Detection --- #

                # Increment frame count and write data to file
                frame_count += 1
                if data_file:
                    data_file.write(f"{frame_count},{current_average_depth},{np.min(depth_map)},{np.max(depth_map)},{is_outlier}\n")

                # Normalize depth map for visualization
                depth_map = cv2.normalize(depth_map, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

                # Resize depth map to match original frame height for concatenation
                # and maintain aspect ratio approximately
                original_height, original_width, _ = frame.shape
                depth_map_height, depth_map_width, _ = depth_map.shape

                if original_height != depth_map_height:
                     scale = original_height / depth_map_height
                     new_width = int(depth_map_width * scale)
                     depth_map = cv2.resize(depth_map, (new_width, original_height))

                # Concatenate original frame and depth map horizontally
                combined_frame = cv2.hconcat([frame, depth_map])

                # Display the combined frame
                cv2.imshow("Webcam Feed (Original vs Depth)", combined_frame)

                # Initialize VideoWriter after the first frame if not already done
                if video_writer is None:
                    # Get dimensions of the combined frame
                    combined_height, combined_width, _ = combined_frame.shape
                    fps = int(capturer.cap.get(cv2.CAP_PROP_FPS))
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Codec for AVI
                    video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (combined_width, combined_height))
                    print(f"Video writer initialized with dimensions: {combined_width}x{combined_height}")

                # Write frame to video
                if video_writer is not None:
                    video_writer.write(combined_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Ensure the data file is closed
            if 'data_file' in locals() and data_file and not data_file.closed:
                data_file.close()
                print(f"Data saved to {data_file_path}")

            capturer.release()
            cv2.destroyAllWindows()
            print("Webcam released.")

            # Release video writer
            video_writer.release()
    else:
        print(f"Failed to start webcam {camera_index}.") 