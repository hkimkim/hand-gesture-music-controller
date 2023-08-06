"""
Heekyung Kim
CS5330 SP 23
Final Project
This script contains code for training the SSD-MobileNetV3 network for our app
"""
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as TF   
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial

from preprocess import ImageDataset, xml_to_csv
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config

# Set backend to cpu
torch.backends.cudnn.enabled = False
device = torch.device("cpu")

DATASET_DIR = "../../Desktop/set3" 
TEST_BATCHSIZE = 5
VAL_BATCHSIZE = 5
TRAIN_BATCHSIZE = 5
num_of_labels = 8

# random seed  
torch.manual_seed(22)

# Tensorboard
writer = SummaryWriter()


# Preprocess the data into train and test dataset
def prepare_dataset(data_dir):

    train_dir = data_dir + "/train"
    val_dir = data_dir + "/val"
    test_dir = data_dir + "/test"

    # Covnert XMl to csv annotation file
    train_df_annotation, labels = xml_to_csv(train_dir)
    val_df_annotation, labels = xml_to_csv(val_dir)
    test_df_annotation, labels = xml_to_csv(test_dir)

    num_of_labels = len(labels) 

    # Create dataset using custom dataset class
    train_dataset = ImageDataset(train_df_annotation, train_dir)
    val_dataset = ImageDataset(val_df_annotation, val_dir)
    test_dataset = ImageDataset(test_df_annotation, test_dir)

    # Create DataLoaders for datasets
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCHSIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCHSIZE, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


# Load model and configure the SSD head
def load_model():

    # Approach 1 - Extract SDDLite head and specify
    # Load mobilenet from model zoo
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

    # Change the head of SSD(=output layer) to match our task
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_of_labels, norm_layer)

    # Approach 2 - use pytorch API
    # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT', progress=True, num_classes=num_of_labels, weights_backbone='DEFAULT', trainable_backbone_layers=0)

    # Transfer model to device
    model = model.to(device)

    return model


# Train the model
def train(network, dataloader, optimizer, log_step, epoch):
    network.train()

    # Keep track of losses for each batch
    current_loss = 0.0
    current_regression_loss = 0.0
    current_classification_loss = 0.0

    # Set tensorboard
    n_iter = 0

    # Train 
    for batch_idx, (images, targets) in enumerate(dataloader):
        _targets = []

        for i in range(len(images)):

            # Extract bounding box and label
            ground_truth_bboxes = targets["boxes"][i]
            ground_truth_labels = targets["labels"][i]

            _targets.append({"boxes": ground_truth_bboxes, "labels": ground_truth_labels})

        images = images.to(device)
    
        # Feed-forward
        optimizer.zero_grad()
        losses, _ = network(images, _targets) # pytorch mobile_netv3 api returns regression and classification losses
        regression_loss = losses["bbox_regression"]
        classification_loss =  losses["classification"]

        # Calcualte loss using multibox loss
        # regression_loss, classification_loss = loss_fn(confidence, location, ground_truth_labels, ground_truth_bboxes)
        loss = regression_loss + classification_loss

        # Back propagation
        loss.backward()
        optimizer.step()

        # Update loss 
        current_loss += loss.item()
        current_regression_loss += regression_loss.item()
        current_classification_loss += classification_loss.item()

        # Logging to tensorboard
        writer.add_scalar('Train/Loss', loss.item(), n_iter)
        writer.add_scalar('Train/Regression_Loss', regression_loss.item(), n_iter)
        writer.add_scalar('Train/Classification Loss', classification_loss.item(), n_iter)

        n_iter += 1

        # For log_step, record the loss
        if batch_idx % log_step == 0:
            avg_loss = current_loss / log_step
            avg_regression_loss = current_regression_loss / log_step
            avg_classification_loss = current_classification_loss / log_step

            print(f"Train set: Epoch: {epoch}, Step: {batch_idx}, " +
                  f"Average Loss: {avg_loss:.4f}, " +
                  f"Average Regression Loss: {avg_regression_loss:.4f}, " + 
                  f"Average Classification Loss: {avg_classification_loss:.4f}"
                )
            
            # Reset loss
            current_loss = 0.0
            current_regression_loss = 0.0
            current_classification_loss = 0.0


# Test the model (validation and test)
def test(network, dataloader, data_name, isTest):
    
    # I refactored code from pytorch torchvision library so taht train() returns prediction and losses
    network.train()

    correct = 0
    num_data_points = 0

    # Keep track of losses for each batch
    current_loss = 0.0
    current_regression_loss = 0.0
    current_classification_loss = 0.0

    for batch_id, (images, targets) in enumerate(dataloader):

        _targets = []

        for i in range(len(images)):

            # Extract bounding box and label
            ground_truth_bboxes = targets["boxes"][i]
            ground_truth_labels = targets["labels"][i]

            _targets.append({"boxes": ground_truth_bboxes, "labels": ground_truth_labels})
            

        with torch.no_grad():
            num_data_points += 1

            losses, detections = network(images, _targets)
            regression_loss = losses["bbox_regression"]
            classification_loss =  losses["classification"]
            loss = regression_loss + classification_loss
        
            # Update loss
            current_loss += loss.item()
            current_regression_loss += regression_loss.item()
            current_classification_loss += classification_loss.item()

            for i, d in enumerate(detections):
                correct += d["labels"][0].eq(_targets[i]["labels"]).sum()
            
        # If test record it to tensorboard
        if isTest:
            print(correct)
            print(100 * correct / (num_data_points * TEST_BATCHSIZE))
            writer.add_scalar(f'Test/Accuracy', 100 * correct / (num_data_points * TEST_BATCHSIZE), batch_id)
        
        
    avg_loss = current_loss / num_data_points
    avg_regression_loss = current_regression_loss / num_data_points
    avg_classification_loss = current_classification_loss / num_data_points
    accuracy = (100. * correct / len(dataloader.dataset))


    print(f"> {data_name} set: " +
          f"Average Loss: {avg_loss:.4f}, " +
          f"Average Regression Loss: {avg_regression_loss:.4f}, " + 
          f"Average Classification Loss: {avg_classification_loss:.4f}, " +
          f"Accuracy: {accuracy:.2f}% \n"
    )
        
    return avg_loss, avg_regression_loss, avg_classification_loss, accuracy


# Save the traiend model
def save_model(model, model_path):
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_path) 
    print(f"Saved PyTorch Model State to {model_path}")


# Main function
def main():

    # Starting/ default value from paper (SSD: Wei Liu)
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 50 
    log_step = 2

    # Load transfered model
    network = load_model()
    
    optimizer = torch.optim.SGD(network.parameters(True), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Prepare dataset
    train_dataloder, val_dataloader, test_dataloader = prepare_dataset(DATASET_DIR)


    # Train + test
    for epoch in range(1, epochs + 1):
        train(network, train_dataloder, optimizer, log_step, epoch)
        avg_loss, avg_regression_loss, avg_classification_loss, accuracy = test(network, val_dataloader, "Val", False)

        writer.add_scalar(f'Val/Loss', avg_loss, epoch)
        writer.add_scalar(f'Val/Regression_Loss', avg_regression_loss, epoch)
        writer.add_scalar(f'Val/Classification Loss', avg_classification_loss, epoch)
        writer.add_scalar(f'Val/Accuracy', accuracy, epoch)

    # Accuracy score from inference
    test(network, test_dataloader, "Test", True)

    # Save model
    save_model(network, "model_test1.pt")
    

if __name__ == "__main__":
    main()



