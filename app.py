"""
Heekyung Kim
CS5330 SP 23
Final Project
This script contains code that runs an application that uses hand gesture classified by trained network
as hot keys to operate YouTube music in real-time.
"""
import os
import cv2 as cv
import pyautogui
from torchvision.models.detection.ssd import SSD
import torch
import torchvision
import torchvision.transforms as transforms
from label_dict import index_to_label

YOUTUBE_MUSIC_PATH = "/Users/heekyungkim/Applications/Chrome\ Apps.localized/YouTube\ Music.app"


# Function to start youtube music app
def start_youtube_music():
    path = YOUTUBE_MUSIC_PATH
    os.system(f"open {path}")



# Function for operating youtube music with hot keys
def command(label):

    match label:
        case 1:
            pyautogui.press(';')
        case 2:
            pyautogui.press(';')
        case 3:
            pyautogui.hotkey('shift', 'n')
        case 4:
            pyautogui.hotkey('shift', 'p')
        case 5:
            pyautogui.hotkey('shift', '+')
        case 6:
            pyautogui.hotkey('shift', '-')
        case 7:
            pyautogui.hotkey('m')


# Main function
def main():

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Load trained model
    model = torch.jit.load('model_test1.pt')

    # Start youtube music
    print("Starting youtube music")
    start_youtube_music()
    
    prevLabel = 0
    same = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame from video stream.")
            break

        key = cv.waitKey(30)

        # Convert the frame to PyTorch tensor
        transform = transforms.ToTensor()
        tensor = transform(frame)

        # Inference
        model.eval()

        _, detection = model([tensor])

        # Extract bounding box, confidence score, and label
        pred_bbox = detection[0]["boxes"][0]
        x_min = int(pred_bbox[0].item())
        y_min = int(pred_bbox[1].item())
        x_max = int(pred_bbox[2].item())
        y_max = int(pred_bbox[3].item())

        pred_score = detection[0]["scores"][0].item()
        pred_label = detection[0]["labels"][0].item()

        label_score = index_to_label[pred_label] + ": " + str(pred_score)

        # Show only predictions with confidence level higher than 0.5
        if (pred_score > 0.5):
            # Draw bounding box
            frame = cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display label and confidence score
            frame = cv.putText(frame, label_score, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

            # Aggregate same commands so that commands are not sent constantly
            if pred_label != prevLabel:
                same = 0
            else:
                if same > 10:
                    # Do action based on prediction label
                    command(pred_label)
                    same = 0
                else:
                    same += 1
                        
            prevLabel = pred_label
            

        cv.imshow('frame', frame)

        if key == ord('q'):
            break
    

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()