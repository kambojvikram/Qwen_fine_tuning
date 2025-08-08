# prepare_data.py - Script to prepare and split dataset for nodes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def prepare_datasets(input_file, num_nodes=4, output_dir="./data"):
    """
    Split dataset across nodes for independent training
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError("Unsupported file format")
    
    # Validate columns
    required_cols = ['prompt', 'question', 'response']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split dataset across nodes
    total_samples = len(df)
    samples_per_node = total_samples // num_nodes
    
    for node_id in range(num_nodes):
        start_idx = node_id * samples_per_node
        if node_id == num_nodes - 1:
            # Last node gets remaining samples
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_node
        
        node_df = df.iloc[start_idx:end_idx]
        
        # Save node-specific dataset
        output_path = os.path.join(output_dir, f"training_data_node_{node_id}.csv")
        node_df.to_csv(output_path, index=False)
        
        print(f"Node {node_id}: {len(node_df)} samples saved to {output_path}")
    
    # Save dataset statistics
    stats = {
        "total_samples": total_samples,
        "num_nodes": num_nodes,
        "samples_per_node": samples_per_node,
        "columns": list(df.columns)
    }
    
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset split complete!")
    print(f"Total samples: {total_samples}")
    print(f"Samples per node: ~{samples_per_node}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset file")
    parser.add_argument("--num_nodes", type=int, default=4, help="Number of nodes")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    
    args = parser.parse_args()
    prepare_datasets(args.input, args.num_nodes, args.output_dir)
