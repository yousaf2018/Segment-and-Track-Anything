import cv2
import argparse
import numpy as np
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker
from tool.transfer_tools import mask2bbox

# Function to parse command-line arguments
def parse_args():
  parser = argparse.ArgumentParser(description="SAM-Track Video Segmentation and Tracking")
  parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
  parser.add_argument("-o", "--output", required=True, help="Path to save the output video")
  parser.add_argument("-m", "--model", default="r50_deaotl", help="SAM-Track model name (default: r50_deaotl)")
  return parser.parse_args()

def main():
  # Parse command-line arguments
  args = parse_args()

  # Load video and initialize capture
  cap = cv2.VideoCapture(args.input)
  ret, frame = cap.read()

  # Define model parameters
  aot_model = args.model
  long_term_mem = 9999
  max_len_long_term = 9999
  sam_gap = 100
  max_obj_num = 255
  points_per_side = 16

  # Initialize SegTracker
  seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
  seg_tracker.restart_tracker()

  # Video processing loop
  while ret:
    # Preprocess frame (if necessary)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Track objects
    predicted_mask, masked_frame = seg_tracker.track(frame)

    # Save output frame
    cv2.imwrite(args.output + "_frame_" + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", masked_frame)

    ret, frame = cap.read()

  # Release resources
  cap.release()

if __name__ == "__main__":
  main()