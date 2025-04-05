# Vaihingen 3D LiDAR Processing for RandLA-Net
# Complete processing pipeline in a single cell for Colab

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import argparse
import time
import datetime
from tqdm import tqdm
import glob
from sklearn.metrics import confusion_matrix

# Helper function for writing PLY files (assuming this is available in your environment)
# Include the helper_ply functions directly
def write_ply(filename, points, fields, field_names=None, triangles=None):
    """
    Write a point cloud to a PLY file.
    :param filename: string - path to the output PLY file
    :param points: list of arrays - list of fields to save
    :param fields: list of strings - list of field names
    :param field_names: list of strings - list of field names for PLY header
    :param triangles: array-like - triangles to save (optional)
    """
    if field_names is None:
        field_names = fields
        
    # List of lines to write
    lines = []
    
    # First line is PLY format
    lines.append('ply')
    
    # ASCII format
    lines.append('format ascii 1.0')
    
    # Points count
    lines.append('element vertex {:d}'.format(points[0].shape[0]))
    
    # Point properties
    for i, prop in enumerate(field_names):
        if prop in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
            lines.append('property float ' + prop)
        elif prop in ['red', 'green', 'blue', 'alpha']:
            lines.append('property uchar ' + prop)
        else:
            lines.append('property float ' + prop)
            
    # Add triangles if provided
    if triangles is not None:
        lines.append('element face {:d}'.format(triangles.shape[0]))
        lines.append('property list uchar int vertex_indices')
        
    # End header
    lines.append('end_header')
    
    # Write all points
    for i in range(points[0].shape[0]):
        line = []
        for j, points_field in enumerate(points):
            if field_names[j] in ['red', 'green', 'blue', 'alpha']:
                line.append('{:d}'.format(points_field[i]))
            else:
                line.append('{:.6f}'.format(points_field[i]))
        lines.append(' '.join(line))
        
    # Write triangles if provided
    if triangles is not None:
        for i in range(triangles.shape[0]):
            line = '3 {:d} {:d} {:d}'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            lines.append(line)
            
    # Write file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

# Add DataProcessing class (assuming it's used by RandLANet)
class DataProcessing:
    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N*3
        :param query_pts: points you want to know the neighbor indices, B*M*3
        :param k: Number of neighbors in knn search
        :return: neighbor_idx: neighboring points indexes, B*M*k
        """
        neighbor_idx = np.zeros((support_pts.shape[0], query_pts.shape[1], k), dtype=np.int32)
        for i in range(support_pts.shape[0]):
            neighbor_idx[i] = DataProcessing.knn_batch(support_pts[i], query_pts[i], k)
        return neighbor_idx

    @staticmethod
    def knn_batch(support_pts, query_pts, k):
        """
        :param support_pts: points you have, N*3
        :param query_pts: points you want to know the neighbor indices, M*3
        :param k: Number of neighbors in knn search
        :return: neighbor_idx: neighboring points indexes, M*k
        """
        from sklearn.neighbors import NearestNeighbors
        neighbor = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(support_pts)
        distances, neighbor_idx = neighbor.kneighbors(query_pts)
        return neighbor_idx

    @staticmethod
    def grid_subsampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid subsampling (method = barycenter for points and features)
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: subsampled points, with features and/or labels depending on the input
        """
        if (features is None) and (labels is None):
            return DataProcessing.grid_subsampling_no_features(points, grid_size, verbose)
        elif labels is None:
            return DataProcessing.grid_subsampling_with_features(points, features, grid_size, verbose)
        elif features is None:
            return DataProcessing.grid_subsampling_with_labels(points, labels, grid_size, verbose)
        else:
            return DataProcessing.grid_subsampling_with_features_and_labels(points, features, labels, grid_size, verbose)

    @staticmethod
    def grid_subsampling_no_features(points, grid_size=0.1, verbose=0):
        """
        Grid subsampling of the points without features
        :param points: [N, 3] ndarray of points
        :param grid_size: grid voxel size
        :param verbose: 1 to display
        :return: [M, 3] ndarray of subsampled points
        """
        # Compute voxel indices for each point
        cloud_voxels = np.floor(points / grid_size).astype(int)
        voxel_indices = np.array([cloud_voxels[:, 0], cloud_voxels[:, 1], cloud_voxels[:, 2]]).T
        
        # Get unique voxels
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        # Compute centers of voxels
        voxel_centers = []
        for i in range(len(counts)):
            voxel_points = points[inverse == i]
            voxel_center = np.mean(voxel_points, axis=0)
            voxel_centers.append(voxel_center)
            
        return np.array(voxel_centers)

    @staticmethod
    def grid_subsampling_with_features(points, features, grid_size=0.1, verbose=0):
        """
        Grid subsampling of the points with features
        :param points: [N, 3] ndarray of points
        :param features: [N, d] ndarray of features
        :param grid_size: grid voxel size
        :param verbose: 1 to display
        :return: [M, 3] ndarray of subsampled points, [M, d] ndarray of subsampled features
        """
        # Compute voxel indices for each point
        cloud_voxels = np.floor(points / grid_size).astype(int)
        voxel_indices = np.array([cloud_voxels[:, 0], cloud_voxels[:, 1], cloud_voxels[:, 2]]).T
        
        # Get unique voxels
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        # Compute centers of voxels and average features
        voxel_centers = []
        voxel_features = []
        for i in range(len(counts)):
            voxel_points = points[inverse == i]
            voxel_point_features = features[inverse == i]
            voxel_center = np.mean(voxel_points, axis=0)
            voxel_feature = np.mean(voxel_point_features, axis=0)
            voxel_centers.append(voxel_center)
            voxel_features.append(voxel_feature)
            
        return np.array(voxel_centers), np.array(voxel_features)

    @staticmethod
    def grid_subsampling_with_labels(points, labels, grid_size=0.1, verbose=0):
        """
        Grid subsampling of the points with labels
        :param points: [N, 3] ndarray of points
        :param labels: [N,] ndarray of labels
        :param grid_size: grid voxel size
        :param verbose: 1 to display
        :return: [M, 3] ndarray of subsampled points, [M,] ndarray of subsampled labels
        """
        # Compute voxel indices for each point
        cloud_voxels = np.floor(points / grid_size).astype(int)
        voxel_indices = np.array([cloud_voxels[:, 0], cloud_voxels[:, 1], cloud_voxels[:, 2]]).T
        
        # Get unique voxels
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        # Compute centers of voxels and most common label
        voxel_centers = []
        voxel_labels = []
        for i in range(len(counts)):
            voxel_points = points[inverse == i]
            voxel_point_labels = labels[inverse == i]
            voxel_center = np.mean(voxel_points, axis=0)
            # Most common label in the voxel
            unique_labels, label_counts = np.unique(voxel_point_labels, return_counts=True)
            voxel_label = unique_labels[np.argmax(label_counts)]
            voxel_centers.append(voxel_center)
            voxel_labels.append(voxel_label)
            
        return np.array(voxel_centers), np.array(voxel_labels)

    @staticmethod
    def grid_subsampling_with_features_and_labels(points, features, labels, grid_size=0.1, verbose=0):
        """
        Grid subsampling of the points with features and labels
        :param points: [N, 3] ndarray of points
        :param features: [N, d] ndarray of features
        :param labels: [N,] ndarray of labels
        :param grid_size: grid voxel size
        :param verbose: 1 to display
        :return: [M, 3] ndarray of subsampled points, [M, d] ndarray of subsampled features, [M,] ndarray of subsampled labels
        """
        # Compute voxel indices for each point
        cloud_voxels = np.floor(points / grid_size).astype(int)
        voxel_indices = np.array([cloud_voxels[:, 0], cloud_voxels[:, 1], cloud_voxels[:, 2]]).T
        
        # Get unique voxels
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        # Compute centers of voxels, average features, and most common label
        voxel_centers = []
        voxel_features = []
        voxel_labels = []
        for i in range(len(counts)):
            voxel_points = points[inverse == i]
            voxel_point_features = features[inverse == i]
            voxel_point_labels = labels[inverse == i]
            voxel_center = np.mean(voxel_points, axis=0)
            voxel_feature = np.mean(voxel_point_features, axis=0)
            # Most common label in the voxel
            unique_labels, label_counts = np.unique(voxel_point_labels, return_counts=True)
            voxel_label = unique_labels[np.argmax(label_counts)]
            voxel_centers.append(voxel_center)
            voxel_features.append(voxel_feature)
            voxel_labels.append(voxel_label)
            
        return np.array(voxel_centers), np.array(voxel_features), np.array(voxel_labels)

# Step 1: Create a basic configuration for the scripts
class Config:
    def __init__(self):
        # Paths
        self.data_dir = '/content/drive/MyDrive/Vaihingen_'
        self.output_dir = '/content/drive/MyDrive/Vaihingen_'
        self.model_path = '/content/drive/MyDrive/Vaihingen_/model'
        self.train_file = 'Vaihingen3D_TRAIN.pts'
        self.test_file = 'Vaihingen3D_EVAL_WITH_REF.pts'
        
        # Training parameters
        self.gpu = 0
        self.mode = 'train'
        self.max_epoch = 100
        self.batch_size = 4
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.decay_step = 200000
        self.decay_rate = 0.7
        self.early_stopping_patience = 10
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

# Step 2: Define the function to load Vaihingen 3D LiDAR data
def load_pts_file(file_path):
    """
    Load .pts file from Vaihingen 3D dataset
    Format: x y z intensity return_number num_returns classification
    
    For Vaihingen3D, classification typically follows:
    0: Powerline
    1: Low vegetation 
    2: Impervious surfaces
    3: Car
    4: Fence/Hedge
    5: Roof
    6: Facade
    7: Shrub
    8: Tree
    9: Undefined (for training file)
    
    Returns:
        points: Nx7 array (x, y, z, intensity, return_number, num_returns, classification)
    """
    print(f"Loading {file_path}...")
    
    # First pass to count lines
    with open(file_path, 'r') as f:
        num_points = sum(1 for _ in f)
    
    # Allocate memory
    points = np.zeros((num_points, 7), dtype=np.float32)
    
    # Second pass to read data
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=num_points)):
            if line.strip():  # Skip empty lines
                values = line.strip().split()
                if len(values) >= 7:  # Ensure we have all fields
                    # x, y, z, intensity, return_number, num_returns, classification
                    points[i, 0] = float(values[0])  # x
                    points[i, 1] = float(values[1])  # y
                    points[i, 2] = float(values[2])  # z
                    points[i, 3] = float(values[3])  # intensity
                    points[i, 4] = float(values[4])  # return_number
                    points[i, 5] = float(values[5])  # num_returns
                    points[i, 6] = float(values[6])  # classification
    
    return points

# Step 3: Process Vaihingen 3D data
def process_vaihingen3d_data(config):
    """Process Vaihingen 3D data without train/val split"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load the training and test data
    train_file = os.path.join(config.data_dir, config.train_file)
    test_file = os.path.join(config.data_dir, config.test_file)
    
    # Load and process the training data
    if os.path.exists(train_file):
        train_points = load_pts_file(train_file)
        
        # Extract features (x, y, z, intensity, return_number, num_returns)
        features = train_points[:, :6]
        
        # Extract labels (classification)
        labels = train_points[:, 6].astype(np.int32)
        
        # Save processed data
        np.save(os.path.join(config.output_dir, 'train_points.npy'), features)
        np.save(os.path.join(config.output_dir, 'train_labels.npy'), labels)
        
        print(f"Training data processed: {features.shape[0]} points")
    else:
        print(f"Warning: Training file {train_file} not found!")
    
    # Load and process the test data
    if os.path.exists(test_file):
        test_points = load_pts_file(test_file)
        
        # Extract features (x, y, z, intensity, return_number, num_returns)
        features = test_points[:, :6]
        
        # Extract labels (classification)
        labels = test_points[:, 6].astype(np.int32)
        
        # Save processed data
        np.save(os.path.join(config.output_dir, 'test_points.npy'), features)
        np.save(os.path.join(config.output_dir, 'test_labels.npy'), labels)
        
        print(f"Test data processed: {features.shape[0]} points")
    else:
        print(f"Warning: Test file {test_file} not found!")
    
    # Create class mapping
    class_names = {
        0: 'Powerline',
        1: 'Low vegetation',
        2: 'Impervious surfaces',
        3: 'Car',
        4: 'Fence/Hedge',
        5: 'Roof',
        6: 'Facade',
        7: 'Shrub',
        8: 'Tree',
        9: 'Undefined'  # May be present in training data
    }
    
    # Save class mapping
    np.save(os.path.join(config.output_dir, 'class_names.npy'), class_names)

# Step 4: Define the Vaihingen 3D dataset class
class Vaihingen3DDataset:
    def __init__(self, dataset_path):
        self.name = 'vaihingen3d'
        self.path = dataset_path
        
        # Load class names and mapping
        class_names_file = os.path.join(dataset_path, 'class_names.npy')
        if os.path.exists(class_names_file):
            self.label_to_names = np.load(class_names_file, allow_pickle=True).item()
        else:
            self.label_to_names = {
                0: 'Powerline',
                1: 'Low vegetation',
                2: 'Impervious surfaces',
                3: 'Car',
                4: 'Fence/Hedge',
                5: 'Roof',
                6: 'Facade',
                7: 'Shrub',
                8: 'Tree',
                9: 'Undefined'
            }
        
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([9])  # Undefined class
        
        # Load the processed data
        self.train_points = np.load(os.path.join(dataset_path, 'train_points.npy'))
        self.train_labels = np.load(os.path.join(dataset_path, 'train_labels.npy'))
        self.test_points = np.load(os.path.join(dataset_path, 'test_points.npy'))
        self.test_labels = np.load(os.path.join(dataset_path, 'test_labels.npy'))
        
        # Calculate mean and std for normalization
        self.mean_xyz = np.mean(self.train_points[:, :3], axis=0)
        self.std_xyz = np.std(self.train_points[:, :3], axis=0)
        self.mean_features = np.mean(self.train_points[:, 3:], axis=0)
        self.std_features = np.std(self.train_points[:, 3:], axis=0)
        
        # Initialize containers for sampling
        self.possibility = {}
        self.min_possibility = {}
        
        # Save calibration data
        calibration_data = {
            'mean_xyz': self.mean_xyz,
            'std_xyz': self.std_xyz,
            'mean_features': self.mean_features,
            'std_features': self.std_features
        }
        with open(os.path.join(dataset_path, 'calibration.pkl'), 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"Vaihingen 3D dataset loaded with {len(self.train_points)} train points")
        print(f"Test split: {len(self.test_points)} points")
    
    def get_batch_gen(self, split, batch_size=8, num_points=40960):
        """Generate batches for training/testing"""
        
        if split == 'train':
            points = self.train_points
            labels = self.train_labels
        elif split == 'test':
            points = self.test_points
            labels = self.test_labels
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Initialize sampling probability
        prob_name = f"{split}_prob"
        if prob_name not in self.possibility:
            self.possibility[prob_name] = np.random.rand(len(points)) * 1e-3
            self.min_possibility[prob_name] = float(np.min(self.possibility[prob_name]))
        
        # Define the batch generator
        def batch_generator():
            while True:
                batch_points = []
                batch_labels = []
                
                for _ in range(batch_size):
                    # Choose point with minimum probability
                    point_idx = np.argmin(self.possibility[prob_name])
                    center_point = points[point_idx, :3]
                    
                    # Get nearby points using KNN
                    distances = np.sum(np.square(points[:, :3] - center_point), axis=1)
                    selected_indices = np.argsort(distances)[:num_points]
                    
                    # Get selected points and labels
                    selected_points = points[selected_indices]
                    selected_labels = labels[selected_indices]
                    
                    # Update probabilities
                    self.possibility[prob_name][selected_indices] += 1
                    self.min_possibility[prob_name] = float(np.min(self.possibility[prob_name]))
                    
                    # Normalize points
                    selected_points = selected_points.astype(np.float32)
                    normalized_points = np.zeros_like(selected_points)
                    normalized_points[:, :3] = (selected_points[:, :3] - self.mean_xyz) / self.std_xyz
                    normalized_points[:, 3:] = (selected_points[:, 3:] - self.mean_features) / self.std_features
                    
                    # Add to batch
                    batch_points.append(normalized_points)
                    batch_labels.append(selected_labels)
                
                # Convert to numpy arrays
                batch_points = np.array(batch_points)
                batch_labels = np.array(batch_labels)
                
                yield batch_points, batch_labels
        
        return batch_generator()

# Step 5: Define the RandLA-Net network architecture
class Network:
    def __init__(self, dataset, config):
        flat_inputs = tf.keras.layers.Input(shape=(None, 6), dtype=tf.float32)
        self.config = config
        self.dataset = dataset
        
        # Network configuration
        self.num_layers = config['num_layers']
        self.num_points = config['num_points'] if 'num_points' in config else 40960
        self.num_classes = config['num_classes']
        self.num_neighbors = config['num_neighbors']
        self.decimation = config['decimation']
        self.grid_size = config['grid_size']
        self.d_out = config['d_out']
        self.ignored_label_inds = config['ignored_label_inds']
        
        # Call the network function
        net = self.network(flat_inputs, self.is_training)
        
        # Output logits
        self.logits = net
        
        # Fully connected segmentation layer
        self.probs = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.probs, axis=-1)
        
    def network(self, inputs, is_training):
        """
        Define the network architecture
        """
        # Initial MLP
        x = inputs
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        
        # Encoder blocks
        features = [x]
        for i in range(self.num_layers):
            # Downsample
            x = self.encoder_block(x, self.d_out[i], self.num_neighbors[i], self.decimation[i], self.grid_size[i])
            features.append(x)
        
        # Feature aggregation
        x = tf.reduce_max(tf.stack(features, axis=0), axis=0)
        
        # MLP for classification
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        
        return x
    
    def encoder_block(self, x, d_out, num_neighbors, decimation, grid_size):
        """
        Encoder block with local spatial encoding and pooling
        """
        # Local spatial encoding
        # Since we can't directly implement the full RandomLA-Net architecture,
        # we'll use a simplified version with dense layers
        x = tf.keras.layers.Dense(d_out // 2, activation='relu')(x)
        x = tf.keras.layers.Dense(d_out, activation='relu')(x)
        
        # Pooling
        # In a real implementation, we would do grid subsampling here
        # For this simplified version, we'll just use max pooling over the feature dimension
        x = tf.keras.layers.MaxPool1D(pool_size=decimation)(x)
        
        return x
    
    def get_loss(self, logits, labels):
        """
        Define the loss function
        """
        # One-hot encode the labels
        one_hot_labels = tf.one_hot(labels, depth=self.num_classes)
        
        # Cross-entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        
        # Mask out the ignored labels
        if len(self.ignored_label_inds) > 0:
            mask = tf.reduce_sum(tf.one_hot(self.ignored_label_inds, depth=self.num_classes), axis=0)
            mask = 1 - mask
            cross_entropy = cross_entropy * mask
        
        # Mean over all points
        loss = tf.reduce_mean(cross_entropy)
        
        return loss
    
    def inference(self, inputs, is_training):
        """
        Network inference
        """
        return self.network(inputs, is_training)

# Step 6: Define the training and testing function
def train_vaihingen3d(config):
    """
    Train the RandLA-Net model on Vaihingen 3D dataset
    """
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    
    # Create dataset
    dataset = Vaihingen3DDataset(config.output_dir)
    
    # Define model parameters for Vaihingen 3D
    model_params = {
        'num_classes': dataset.num_classes,
        'd_out': [16, 64, 128, 256],             # Hidden dimensions
        'num_neighbors': [16, 16, 16, 16],       # KNN
        'decimation': [4, 4, 4, 4],              # Grid subsampling rate
        'grid_size': [0.1, 0.2, 0.4, 0.8],       # Grid size for subsampling (in meters)
        'num_layers': 4,                         # Number of layers
        'ignored_label_inds': dataset.ignored_labels
    }
    
    # Define the model
    with tf.Graph().as_default():
        # Define placeholders
        pl_points = tf.placeholder(tf.float32, shape=(None, None, 6))
        pl_labels = tf.placeholder(tf.int32, shape=(None, None))
        pl_is_training = tf.placeholder(tf.bool, shape=())
        
        # Define model
        model = Network(dataset, model_params)
        logits = model.inference(pl_points, pl_is_training)
        
        # Define loss
        loss = model.get_loss(logits, pl_labels)
        
        # Define optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            config.learning_rate,
            global_step,
            config.decay_step,
            config.decay_rate,
            staircase=True
        )
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        # Define accuracy
        predictions = tf.argmax(logits, axis=2)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(pl_labels, tf.int64)), tf.float32))
        
        # Create saver
        saver = tf.train.Saver(max_to_keep=5)
        
        # Create session
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        config_tf.allow_soft_placement = True
        sess = tf.Session(config=config_tf)
        sess.run(tf.global_variables_initializer())
        
        # Create log directory
        log_dir = os.path.join(config.model_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create summary writer
        train_summary = tf.summary.scalar('train_loss', loss)
        train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
        train_summary_op = tf.summary.merge([train_summary, train_acc_summary])
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
        # Training loop
        print("Start training...")
        best_val_loss = float('inf')
        patience_count = 0
        
        # Get batch generators
        train_gen = dataset.get_batch_gen('train', batch_size=config.batch_size)
        test_gen = dataset.get_batch_gen('test', batch_size=config.batch_size)
        
        for epoch in range(config.max_epoch):
            # Train
            epoch_loss = 0
            epoch_acc = 0
            step_count = 0
            
            for i in range(100):  # Process 100 batches per epoch
                batch_points, batch_labels = next(train_gen)
                
                # Run training step
                _, loss_val, acc_val, summary, step = sess.run(
                    [train_op, loss, accuracy, train_summary_op, global_step],
                    feed_dict={
                        pl_points: batch_points,
                        pl_labels: batch_labels,
                        pl_is_training: True
                    }
                )
                
                # Update metrics
                epoch_loss += loss_val
                epoch_acc += acc_val
                step_count += 1
                
                # Write summary
                summary_writer.add_summary(summary, step)
            
            # Calculate average metrics
            epoch_loss /= step_count
            epoch_acc /= step_count
            
            print(f"Epoch {epoch+1}/{config.max_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
            # Validation
            val_loss = 0
            val_acc = 0
            val_count = 0
            
            for i in range(20):  # Process 20 batches for validation
                batch_points, batch_labels = next(test_gen)
                
                # Run forward pass
                loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={
                        pl_points: batch_points,
                        pl_labels: batch_labels,
                        pl_is_training: False
                    }
                )
                
                # Update metrics
                val_loss += loss_val
                val_acc += acc_val
                val_count += 1
            
            # Calculate average validation metrics
            val_loss /= val_count
            val_acc /= val_count
            
            print(f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saver.save(sess, os.path.join(config.model_path, 'model.ckpt'))
                patience_count = 0
                print("Model saved!")
            else:
                patience_count += 1
                
            # Early stopping
            if patience_count >= config.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print("Training completed!")

# Step 7: Define the evaluation function
def evaluate_vaihingen3d(config):
    """
    Evaluate the trained model on Vaihingen 3D test data
    """
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    
    # Create dataset
    dataset = Vaihingen3DDataset(config.output_dir)
    
    # Define model parameters
    model_params = {
        'num_classes': dataset.num_classes,
        'd_out': [16, 64, 128, 256],
        'num_neighbors': [16, 16, 16, 16],
        'decimation': [4, 4, 4, 4],
        'grid_size': [0.1, 0.2, 0.4, 0.8],
        'num_layers': 4,
        'ignored_label_inds': dataset.ignored_labels
    }
    
    # Load test data
    test_points = np.load(os.path.join(config.output_dir, 'test_points.npy'))
    test_labels = np.load(os.path.join(config.output_dir, 'test_labels.npy'))
    
    # Normalize test points
    test_points_normalized = np.zeros_like(test_points)
    test_points_normalized[:, :3] = (test_points[:, :3] - dataset.mean_xyz) / dataset.std_xyz
    test_points_normalized[:, 3:] = (test_points[:, 3:] - dataset.mean_features) / dataset.std_features
    
    # Define graph
    with tf.Graph().as_default():
        # Define placeholders
        pl_points = tf.placeholder(tf.float32, shape=(None, None, 6))
        pl_is_training = tf.placeholder(tf.bool, shape=())
        
        # Define model
        model = Network(dataset, model_params)
        logits = model.inference(pl_points, pl_is_training)
        predictions = tf.argmax(logits, axis=2)
        
        # Create session
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        config_tf.allow_soft_placement = True
        sess = tf.Session(config=config_tf)
        
        # Load model
        saver = tf.train.Saver()
        model_path = os.path.join(config.model_path, 'model.ckpt')
        saver.restore(sess, model_path)
        print(f"Model restored from {model_path}")
        
        # Evaluate on test data
        print("Starting evaluation...")
        
        # Process data in batches
        batch_size = 4
        num_points = 40960
        all_preds = []
        
        for i in range(0, len(test_points), num_points):
            end_idx = min(i + num_points, len(test_points))
            batch_points = test_points_normalized[i:end_idx]
            
            # Ensure batch has proper shape
            if len(batch_points) < num_points:
                pad_size = num_points - len(batch_points)
                batch_points = np.vstack([batch_points, np.zeros((pad_size, 6), dtype=np.float32)])
            
            batch_points = batch_points.reshape(1, -1, 6)
            
            # Run prediction
            pred = sess.run(
                predictions,
                feed_dict={
                    pl_points: batch_points,
                    pl_is_training: False
                }
            )
            
            # Only keep predictions for original points
            if end_idx - i < num_points:
                pred = pred[:, :(end_idx - i)]
            
            all_preds.append(pred.reshape(-1))
        
        # Combine all predictions
        all_preds = np.concatenate(all_preds)[:len(test_labels)]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(test_labels, all_preds, labels=range(dataset.num_classes))
        
        # Calculate per-class metrics
        class_iou = np.zeros(dataset.num_classes)
        for i in range(dataset.num_classes):
            # IoU = TP / (TP + FP + FN)
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            if tp + fp + fn > 0:
                class_iou[i] = tp / (tp + fp + fn)
        
        # Calculate overall accuracy and mean IoU
        overall_acc = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        mean_iou = np.mean(class_iou)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print("\nPer-Class IoU:")
        for i in range(dataset.num_classes):
            class_name = dataset.label_to_names[dataset.label_values[i]]
            print(f"{class_name}: {class_iou[i]:.4f}")
        
        # Save results as PLY file for visualization
        os.makedirs(os.path.join(config.output_dir, 'results'), exist_ok=True)
        result_file = os.path.join(config.output_dir, 'results', 'vaihingen3d_results.ply')
        
        # Prepare data for PLY file
        points_xyz = test_points[:, :3]
        
        # Define colors for each class
        class_colors = {
            0: [255, 0, 0],      # Powerline: Red
            1: [0, 255, 0],      # Low vegetation: Green
            2: [128, 128, 128],  # Impervious surfaces: Gray
            3: [255, 255, 0],    # Car: Yellow
            4: [0, 255, 255],    # Fence/Hedge: Cyan
            5: [255, 0, 255],    # Roof: Magenta
            6: [165, 42, 42],    # Facade: Brown
            7: [0, 128, 0],      # Shrub: Dark Green
            8: [0, 0, 255],      # Tree: Blue
            9: [0, 0, 0]         # Undefined: Black
        }
        
        # Map predictions to colors
        pred_colors = np.zeros((len(all_preds), 3), dtype=np.uint8)
        for i, pred in enumerate(all_preds):
            pred_colors[i] = class_colors[int(pred)]
        
        # Map ground truth to colors
        gt_colors = np.zeros((len(test_labels), 3), dtype=np.uint8)
        for i, label in enumerate(test_labels):
            gt_colors[i] = class_colors[int(label)]
        
        # Save ground truth and prediction PLY files
        write_ply(
            os.path.join(config.output_dir, 'results', 'vaihingen3d_gt.ply'),
            [points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], gt_colors[:, 0], gt_colors[:, 1], gt_colors[:, 2]],
            ['x', 'y', 'z', 'red', 'green', 'blue']
        )
        
        write_ply(
            os.path.join(config.output_dir, 'results', 'vaihingen3d_pred.ply'),
            [points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], pred_colors[:, 0], pred_colors[:, 1], pred_colors[:, 2]],
            ['x', 'y', 'z', 'red', 'green', 'blue']
        )
        
        print(f"Results saved to {os.path.join(config.output_dir, 'results')}")

# Main execution
if __name__ == "__main__":
    # Create config
    config = Config()
    
    # Process data
    process_vaihingen3d_data(config)
    
    # Train model
    if config.mode == 'train':
        train_vaihingen3d(config)
    
    # Evaluate model
    if config.mode == 'test':
        evaluate_vaihingen3d(config)