import cv2
import os

video_folder = '/data2/huyaojun/data/SurtoolDataset/video_clips'
output_folder = '../../data/SurtoolDataset/images'

save_intervene = 60 * 15

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the video files in the video folder
for filename in os.listdir(video_folder):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        # Open the video file
        cap = cv2.VideoCapture(os.path.join(video_folder, filename))
        count = 0

        # Read frames from the video and save them as images
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save every nth frame (in this example, every 10th frame)
            if count % save_intervene == 0:
                output_filename = os.path.splitext(filename)[0] + f'-frame{count}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, frame)
                print("saved:" + output_filename)

            count += 1

        cap.release()