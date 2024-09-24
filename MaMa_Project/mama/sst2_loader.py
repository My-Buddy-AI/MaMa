import datasets

def load_sst2():
    """Load and return SST-2 dataset."""
    dataset = datasets.load_dataset('glue', 'sst2')
    return dataset
