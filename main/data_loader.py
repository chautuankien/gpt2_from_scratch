import h5py
import numpy as np
import torch

from config.config import GPTConfig

class BatchIterator:
    def __init__(self, config: GPTConfig, device: str):
        self.config = config
        self.device = device
        self.file_handle = None
        self.dataset = None
        self.setup_dataset()
        
    def setup_dataset(self):
        """Initialize the HDF5 file and dataset"""
        self.file_handle = h5py.File(self.config.TRAIN_PATH, 'r')
        self.dataset = self.file_handle["dataset"]
        
        # Calculate dataset properties
        self.dataset_size = self.dataset.shape[0]
        self.n_examples = (self.dataset_size - 1) // self.config.BLOCK_SIZE
        # self.n_examples = (self.dataset_size - 1) // (self.config.BLOCK_SIZE * self.config.BATCH_SIZE)
        
        # Initialize tracking variables
        self.example_idxs = np.arange(self.n_examples)
        np.random.shuffle(self.example_idxs)
        self.epochs = 0
        self.counter = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Check if we need to start a new epoch
        if self.counter + self.config.BATCH_SIZE > self.n_examples:
            np.random.shuffle(self.example_idxs)
            self.counter = 0
            print(f"Finished epoch {self.epochs}")
            self.epochs += 1
        
        # Get batch indices
        random_indices = self.example_idxs[self.counter: self.counter + self.config.BATCH_SIZE] #* self.config.BLOCK_SIZE
        
        # Load data
        random_samples = torch.tensor(np.array([
            self.dataset[idx:idx + self.config.BLOCK_SIZE + 1] 
            for idx in random_indices
        ]))
        
        # Prepare input and target tensors
        xb = random_samples[:, :-1].to(self.device)
        yb = random_samples[:, 1:].to(self.device)
        
        self.counter += self.config.BATCH_SIZE
        
        return xb, yb
    
    def close(self):
        """Properly close the HDF5 file"""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
    
    def __del__(self):
        """Destructor to ensure file is closed"""
        self.close()

def get_batch_iterator(config: GPTConfig, device: str):
    """
    Creates an iterator that yields batches of input and target sequences from an HDF5 dataset.
    Args:
        config (GPTConfig): Configuration object
        device (str): Device identifier (e.g., 'cpu' or 'cuda') to which tensors will be moved.   
    Returns:
        BatchDataLoader: An iterator that yields batches of input and target sequences.
    """
    return BatchDataLoader(config, device)


