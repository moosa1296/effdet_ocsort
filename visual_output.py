import cv2
from collections import defaultdict

def read_tracking_results(file_path):
    tracking_results = {}
    with open(file_path, 'r') as file:
        # lines = file.readlines()
        # print("First few lines in the file:", lines[:5])
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            x_min, y_min, x_max, y_max, track_id, _, _, frame_number = map(float, parts[:8])
            frame_number = int(frame_number)
            track_id = int(track_id)

            if frame_number not in tracking_results:
                tracking_results[frame_number] = []

            # x_max = x_min + width
            # y_max = y_min + height
            tracking_results[frame_number].append((x_min, y_min, x_max, y_max, track_id))

    # print("Total frames with tracking data:", len(tracking_results))
    return tracking_results


# Initialize tracking history
track_history = defaultdict(list)

# Read tracking results
tracking_results = read_tracking_results('/home/user-1/results/tracking_results.txt')
# print("Sample tracking data:", list(tracking_results.items())[:5]) 
# Open the video
cap = cv2.VideoCapture('/home/user-1/tracking/val/videos/avd13_cam3_20220302140230_20220302140730.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
output_path = '/home/user-1/results/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

def rescale_bbox(x_min, y_min, x_max, y_max, scale_x, scale_y):
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y

original_width, original_height = 2688, 1520  # Replace with the resolution used for tracking
current_width, current_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale_x = current_width / original_width
scale_y = current_height / original_height


current_frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # print("Processing frame:", current_frame_number)
    # Draw tracking results and update tracking history
    if current_frame_number in tracking_results:
        for bbox_data in tracking_results[current_frame_number]:
            # x_min, y_min, x_max, y_max, track_id = bbox_data
            x_min, y_min, x_max, y_max, track_id = bbox_data
            x_min, y_min, x_max, y_max = rescale_bbox(x_min, y_min, x_max, y_max, scale_x, scale_y)
            color = (int(track_id * 25) % 256, int(track_id * 50) % 256, int(track_id * 75) % 256)  # Generate a color based on track_id
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 6)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)
    
    # Writing the frame into the output video file
    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

    current_frame_number += 1
    # Draw tracking paths
    # frame = draw_tracking_path(frame, track_history)


cap.release()
out.release()
cv2.destroyAllWindows()

