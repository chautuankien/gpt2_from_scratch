import os
import h5py
import tiktoken
from datasets import load_dataset

def dataset_loader(dataset_name, config_name, split=None):
    return load_dataset(dataset_name, config_name, split=split, cache_dir="./my_cache")


def process_dataset(dataset, output_file, tokenizer_name="gpt2"):
    # Initialize tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    enc = tiktoken.get_encoding(tokenizer_name)

    # Create output dir if doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "w") as f:
        # Create an expandable dataset named 'dataset' in the HDF5 file
        h5py_dataset = f.create_dataset("dataset", (0,), maxshape=(None,), dtype='i')
        start_index = 0
        total_documents = 0

        for example in dataset:
            # Find the key in dataset to extract text content
            key_field = None
            for key, value in example.items():
                key_field = key
                break

            if key_field:
                text = example[key_field]
            else:
                print(f"Could not find the key field in dataset")
                continue
            
            # Append the end-of-text token and encode
            text_with_end = text + "<|endoftext|>"
            encoded = enc.encode(text_with_end, allowed_special={"<|endoftext|>"})
            encoded_len = len(encoded)

            # Calculate the end index for the new tokens
            end_index = start_index + encoded_len

            # Expand the dataset size and store the encoded tokens
            h5py_dataset.resize(h5py_dataset.shape[0] + encoded_len, axis=0)
            h5py_dataset[start_index:end_index] = encoded

            # Update the start index for the next batch of tokens
            start_index = end_index
            total_documents += 1
        
        print("Processing complete!")
        print(f"Total documents processed: {total_documents}")
        print(f"Total tokens: {start_index}")
        print(f"Output saved to: {output_file}")

