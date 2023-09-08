"""
Heekyung Kim
CS5330 SP 23
Final Project
This script contains code to collect image dataset
"""

import cv2 as cv
import os


labels = ["play", "pause", "next", "prev", "thumbsup", "thumbsdown", "mute"]
TRAIN_NUM_DATASET = 10
VAL_NUM_DATASET = TRAIN_NUM_DATASET / 2
TEST_NUM_DATASET = 10

TRAIN_IMG_DIR_PATH = "./dataset/train"
VAL_IMG_DIR_PATH = "./dataset/val"
TEST_IMG_DIR_PATH = "./dataset/test"

num_dataset = 0

# Function that creates directory to save the dataset
def create_directory(root_dir):

    for label in labels:
        path = os.path.join(root_dir, label)

        if not os.path.exists(path):
            os.makedirs(path)


def main():

    root_dir = TRAIN_IMG_DIR_PATH

    # Ask fo input
    print("Press 0 for Train, Press 1 for Validation, Press 2 for Test ")
    dataset = int(input())

    # Create directory to save the training dataset
    match (dataset):
        case 0:
            root_dir = TRAIN_IMG_DIR_PATH
            num_dataset = TRAIN_NUM_DATASET
        
        case 1:
            root_dir = VAL_IMG_DIR_PATH
            num_dataset = VAL_NUM_DATASET

        case 2:
            root_dir = TEST_IMG_DIR_PATH
            num_dataset = TEST_NUM_DATASET
        case _:
            print("wront input!")
            exit(-1)


    print("Creating directories..")
    create_directory(root_dir)
    print("Done creating directories!")

    print("Let's start!")
    cap = cv.VideoCapture(0)
    label_idx = 0
    img_num = 1
    
    print("First pose: ", labels[label_idx])

    while True:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
 
        ret, frame = cap.read()

        key = cv.waitKey(1)
        
        # If all the labels has been created quit the app
        if (label_idx > len(labels) - 1):
            break

        # Create path to save the image
        img_path = os.path.join(root_dir, labels[label_idx], f"{labels[label_idx]}_{img_num}.png" )

        cv.imshow('frame', frame)

        # if s key is pressed, save the image
        if key == ord('s'):
            print(f"Current label: {labels[label_idx]}")

            cv.imwrite(img_path, frame)
            
            print(f"Collected: {img_num} / {num_dataset}")
            print(f"Saved to {img_path}")
            img_num += 1

            # Check if all the images has been collected for a label
            if (img_num > num_dataset):
                label_idx +=1
                img_num = 1

                if (label_idx < len(labels)):
                    print("Next pose: ", labels[label_idx])


        if key == ord('q'):
            break
    

    print("Complete!")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


