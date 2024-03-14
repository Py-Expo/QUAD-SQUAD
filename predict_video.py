import os
import cv2
from ultralytics import YOLO
import time

def main():
    """Main function for cow detection in video."""

    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, 'cow.mp4')  # Replace with your video path
    video_path_out = '{}_out.mp4'.format(video_path)

    # Check if video exists
    if not os.path.exists(video_path):
        print("Error: Video file not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video:", video_path)
        return

    # Read first frame for shape information
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame from video:", video_path)
        cap.release()
        return

    H, W, _ = frame.shape

    # Load YOLO model
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')  # Replace with your model path
    model = YOLO(model_path)

    # Set detection threshold
    threshold = 0.5

    # Create video writer
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    start_time = time.time()  # Start time for FPS calculation

    frame_count = 0
    cow_count = 0
    cow_ids = {}  # Dictionary to track detected cows
    total_cow_count = 0  # Ground truth cow count
    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold and results.names[int(class_id)] == 'cow':
                # Generate a unique cow ID based on the coordinates
                cow_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                if cow_id not in cow_ids:
                    cow_ids[cow_id] = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, f'Cow {cow_count}', (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    cow_count += 1

        # Update total cow count
        total_cow_count += ground_truth_cow_count_for_frame(frame_count)

        out.write(frame)
        ret, frame = cap.read()

        frame_count += 1

    # Calculate and display FPS
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"Average FPS: {fps:.2f}")

    
    print(f"Total number of cows: {cow_count}")

    if total_cow_count > 0:
        accuracy = cow_count / total_cow_count * 100
        print(f"Count accuracy: {accuracy:.2f}%")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def ground_truth_cow_count_for_frame(frame_number):
    """Dummy function to provide ground truth cow count for each frame.
    Replace this with your actual ground truth counting method."""
    # Example: return the frame number as the ground truth cow count
    return frame_number

if __name__ == "__main__":
    main()
