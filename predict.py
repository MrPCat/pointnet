import torch
import numpy as np
import laspy
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
model_path = r"C:\Users\faars\Downloads\modelnet40ply2048-train-pointnet++.pth"
new_points_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITH_REF.pts"

# Define PointNet++ Model Architecture
# This is a simplified version - you may need to adjust to match your pre-trained model architecture
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        # Simplified forward pass (placeholder)
        # In a real implementation, this would include sampling, grouping, and PointNet operations
        B, C, N = xyz.shape
        new_xyz = xyz[:, :, :self.npoint] if self.npoint is not None else xyz
        new_points = points.unsqueeze(-1).permute(0, 3, 1, 2) if points is not None else None
        
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        new_points = torch.max(new_points, dim=-1, keepdim=False)[0]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        # Simplified forward pass (placeholder)
        B, C, N = xyz.shape
        new_xyz = xyz[:, :, :self.npoint]
        
        new_points_list = []
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            new_points = points.unsqueeze(-1).permute(0, 3, 1, 2) if points is not None else None
            
            # Apply MLP
            for j, conv in enumerate(self.conv_blocks[i]):
                bn = self.bn_blocks[i][j]
                new_points = F.relu(bn(conv(new_points)))
                
            new_points = torch.max(new_points, dim=-1, keepdim=False)[0]
            new_points_list.append(new_points)
            
        new_points = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points


class PointNet2ClsMsg(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x


# Custom Dataset for Vaihingen point cloud
class VaihingenDataset(Dataset):
    def __init__(self, file_path, num_points=2048):
        self.num_points = num_points
        
        # Check if the file is .las or .pts
        if file_path.endswith('.las') or file_path.endswith('.laz'):
            self.data = self.load_las_file(file_path)
        elif file_path.endswith('.pts'):
            self.data = self.load_pts_file(file_path)
        else:
            raise ValueError("Unsupported file format. Use .las, .laz, or .pts")
        
        print(f"Loaded point cloud with {self.data['xyz'].shape[0]} points")
        
    def load_las_file(self, file_path):
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        
        # If RGB exists
        try:
            rgb = np.vstack((las.red, las.green, las.blue)).transpose()
        except:
            rgb = np.zeros((xyz.shape[0], 3))
        
        # If classification exists
        try:
            classes = las.classification
        except:
            classes = np.zeros(xyz.shape[0])
        
        return {
            'xyz': xyz,
            'rgb': rgb,
            'labels': classes
        }
    
    def load_pts_file(self, file_path):
        # Custom PTS file reader
        data = {}
        
        try:
            # Try automatic loading first
            pts_data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
            xyz = pts_data[:, :3]
            
            # Extract RGB if available (columns 3-5)
            if pts_data.shape[1] > 5:
                rgb = pts_data[:, 3:6]
            else:
                rgb = np.zeros((xyz.shape[0], 3))
            
            # Extract classification if available (usually the last column)
            if pts_data.shape[1] > 9:  # Assuming classification is in column 10
                classes = pts_data[:, 9]
            else:
                classes = np.zeros(xyz.shape[0])
                
        except:
            # If automatic loading fails, try manual parsing
            print("Automatic loading failed. Trying manual parsing...")
            
            # Read the file content
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header if it exists
            start_idx = 1 if not lines[0][0].isdigit() else 0
            
            # Parse data lines
            xyz_list = []
            rgb_list = []
            class_list = []
            
            for line in lines[start_idx:]:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        xyz_list.append([x, y, z])
                        
                        # RGB if available
                        if len(parts) >= 6:
                            r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                            rgb_list.append([r, g, b])
                        
                        # Classification if available
                        if len(parts) >= 10:
                            cls = int(float(parts[9]))
                            class_list.append(cls)
                    except:
                        continue
            
            xyz = np.array(xyz_list)
            
            if rgb_list:
                rgb = np.array(rgb_list)
            else:
                rgb = np.zeros((xyz.shape[0], 3))
                
            if class_list:
                classes = np.array(class_list)
            else:
                classes = np.zeros(xyz.shape[0])
        
        return {
            'xyz': xyz,
            'rgb': rgb,
            'labels': classes
        }
    
    def __len__(self):
        # Calculate number of samples based on the total points and points per sample
        # For simplicity, we'll create one sample per 2048 points
        return max(1, self.data['xyz'].shape[0] // self.num_points)
    
    def __getitem__(self, idx):
        # Extract a subset of points for this sample
        start_idx = idx * self.num_points
        end_idx = min((idx + 1) * self.num_points, self.data['xyz'].shape[0])
        
        # If we don't have enough points, repeat some points
        if end_idx - start_idx < self.num_points:
            indices = np.concatenate([
                np.arange(start_idx, end_idx),
                np.random.choice(np.arange(start_idx, end_idx), self.num_points - (end_idx - start_idx))
            ])
        else:
            indices = np.arange(start_idx, end_idx)
        
        # Get points and labels
        xyz = self.data['xyz'][indices]
        labels = self.data['labels'][indices]
        
        # Center the points
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        # Normalize to unit sphere
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        
        # Create the features (for simplicity, just using XYZ coordinates)
        # You could also incorporate RGB or other features here
        features = xyz.copy()
        
        # Convert to PyTorch tensor
        xyz_tensor = torch.FloatTensor(xyz).transpose(0, 1)  # [3, N]
        features_tensor = torch.FloatTensor(features).transpose(0, 1)  # [3, N]
        labels_tensor = torch.LongTensor(labels)
        
        return xyz_tensor, features_tensor, labels_tensor

# Function to load the pre-trained model
def load_model(model_path, num_classes):
    model = PointNet2ClsMsg(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try different ways to load the state dict based on how the model was saved
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Remove "module." prefix if it exists (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            # Assume the checkpoint is directly the state dict
            # Remove "module." prefix if it exists (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
            
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing with random weights.")
    
    model = model.to(device)
    model.eval()
    return model

# Function to evaluate the model
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (points, features, target) in enumerate(dataloader):
            points, features, target = points.to(device), features.to(device), target.to(device)
            
            # Get predictions
            pred = model(points)
            pred_choice = pred.data.max(1)[1]
            
            # Store predictions and labels
            all_preds.append(pred_choice.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Evaluated {batch_idx} batches.")
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return all_preds, all_labels

# Function to visualize results
def visualize_results(preds, labels, class_names=None):
    # Convert class indices to names if provided
    if class_names is None:
        class_names = [str(i) for i in range(max(np.max(preds), np.max(labels)) + 1)]
        
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

# Main evaluation function
def main():
    # Create dataset and dataloader
    dataset = VaihingenDataset(new_points_path, num_points=2048)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Determine number of classes from the dataset
    # ModelNet40 has 40 classes, but your data might have different classes
    num_classes = 40  # Default for ModelNet40
    
    # You can also determine the number of classes from your data:
    unique_classes = np.unique(dataset.data['labels'])
    actual_num_classes = len(unique_classes)
    print(f"Found {actual_num_classes} unique classes in the data")
    
    # Load the model
    model = load_model(model_path, num_classes)
    
    # Evaluate the model
    print("Evaluating model...")
    predictions, labels = evaluate_model(model, dataloader)
    
    # Define class names if you have them
    # For Vaihingen dataset, you might want to use the actual class names
    # Example: class_names = ["ground", "building", "tree", "car", ...]
    class_names = None  # Set to None to use numeric class IDs
    
    # Visualize results
    visualize_results(predictions, labels, class_names)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()