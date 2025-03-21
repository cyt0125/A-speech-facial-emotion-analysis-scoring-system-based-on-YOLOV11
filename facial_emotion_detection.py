
import cv2
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import time

video_path = "./video/4.Trump.mp4"
def capture_camera():
    # Turn on the camera or video
    cap = cv2.VideoCapture(video_path)  # 0 represents default camera
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    return cap

# Load pre trained YOLOv11 model
model = YOLO("best1.pt")  # Replace with your YOLOv11 model path

def detect_emotions(frame):
    # Using YOLOv11 model for inference
    results = model(frame)

    # Analysis results
    emotions = []
    for result in results:
        boxes = result.boxes  # Get detection box
        for box in boxes:
            cls_id = int(box.cls)  # Category ID
            cls_name = model.names[cls_id]  # Category name (emotion)
            emotions.append(cls_name)
    return emotions

def emotion_to_score(emotion):
    # Define the mapping from facial expressions to scores
    emotion_scores = {
        'happy': 10,
        'neutral': 5,
        'sad': 3,
        'angry': 1,
        'fear': 2,
        'disgust': 1,
        'surprise': 7
    }
    return emotion_scores.get(emotion, 5)  # If the expression is not in the dictionary, return 5 points

def calculate_average_score(emotions):
    if not emotions:
        return 5  # If no emotion is detected, return 5 points
    scores = [emotion_to_score(emotion) for emotion in emotions]
    return sum(scores) / len(scores)


def main(): # 名字重了就改成别的
    cap = cv2.VideoCapture(video_path)  # Turn on the camera
    interval = 1  # Record every n seconds
    total_scores = []


    try:
        while True:
            ret, frame = cap.read()  # Read a frame
            if not ret:
                print("Unable to read camera frames")
                break

            # Display camera image
            cv2.imshow("Camera", frame)

            # Record facial emotion results every intervals
            if int(time.time()) % interval == 0:
                emotions = detect_emotions(frame)  # Detecting facial expressions
                avg_score = calculate_average_score(emotions)  # Calculate the average score
                total_scores.append(avg_score)  # Record scores

                print(f"Current Emotion: {emotions}, Average Score: {avg_score}")

            # Press the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

        # calculate average score
        final_avg = sum(total_scores) / len(total_scores) if total_scores else 0
        print(f"Speaker's average facial emotion score: {final_avg}")
        if final_avg > 8:
            print("The facial expression score is high, We hope you can continue to maintain it.")
        elif final_avg < 5.5:
            print("The facial expression score is low, We hope you can make some adjustments,")
            print("such as smiling more during speeches.")
        analyze_scores(total_scores)  # 这个函数是画图，如果太卡可以不要

# data analysis
def analyze_scores(total_scores):
    df = pd.DataFrame(total_scores, columns=['score'])
    print(df.describe())
    plt.plot(df['score'], label='Emotion score')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Speaker emotion score changes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()