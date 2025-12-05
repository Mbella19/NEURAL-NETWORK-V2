
import numpy as np
import json
import os
import sys

def convert_npz_to_json(npz_path, output_dir):
    try:
        print(f"Loading {npz_path}...")
        data = np.load(npz_path)
        
        # Target size reduction strategy
        # Total rows ~97k. Stride 10 => ~9.7k rows.
        # JSON size approx: 48MB / 2 = 24MB.
        STRIDE = 10
        DECIMALS = 4
        
        output_data = {}
        processed_keys = []
        
        for key in data.files:
            item = data[key]
            
            # Handle array data
            if isinstance(item, np.ndarray) and item.ndim > 0 and len(item) > 1000:
                original_shape = item.shape
                # Subsample large arrays
                subsampled = item[::STRIDE]
                
                # Round floats
                if np.issubdtype(subsampled.dtype, np.floating):
                    subsampled = np.round(subsampled, DECIMALS)
                
                output_data[key] = subsampled.tolist()
                print(f"Key: {key}, Orig: {original_shape} -> Subsampled: {subsampled.shape}")
            elif isinstance(item, np.ndarray) and item.ndim == 0:
                 # Scalar
                 val = item.item()
                 if isinstance(val, (float, np.floating)):
                     val = round(val, DECIMALS)
                 output_data[key] = val
                 print(f"Key: {key}, Scalar: {val}")
            else:
                # Small arrays or other types
                if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.floating):
                    item = np.round(item, DECIMALS)
                output_data[key] = item.tolist() if isinstance(item, np.ndarray) else item
                print(f"Key: {key}, Small/Other: {np.shape(item)}")

        # Add metadata about subsampling
        output_data['_meta'] = {
            'stride': STRIDE,
            'decimals': DECIMALS,
            'original_file': os.path.basename(npz_path),
            'note': 'Subsampled for reduced file size'
        }
            
        # Create output filename
        filename = os.path.basename(npz_path).replace('.npz', '.json')
        output_path = os.path.join(output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, separators=(',', ':')) # Minimal separators to save space
            
        print("Conversion complete!")
        
        # Check size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    npz_file = "/Users/gervaciusjr/Desktop/AI Trading Bot/CLAUDE 4/data/processed/analyst_cache.npz"
    # Using the exact path provided by the user
    target_dir = "/Users/gervaciusjr/Downloads/converted_txt_files (1)"
    
    convert_npz_to_json(npz_file, target_dir)
