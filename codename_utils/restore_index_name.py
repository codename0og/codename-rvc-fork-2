import faiss
import os

def get_index_info(index_path):
    index = faiss.read_index(index_path)
    
    index_type = type(index).__name__
    num_vectors = index.ntotal
    nlist = index.nlist if hasattr(index, 'nlist') else 'N/A'
    nprobe = index.nprobe if hasattr(index, 'nprobe') else 'N/A'
    
    return index_type, num_vectors, nlist, nprobe

def main():
    index_path = input("Enter the path to / name of the index file: ").strip()
    
    if not os.path.isfile(index_path):
        print(f"Error: The file '{index_path}' does not exist.")
        return
    
    custom_name = input("Enter a custom name: ").strip()
    
    index_type, num_vectors, nlist, nprobe = get_index_info(index_path)
    
    if index_type == 'IndexIVFFlat':
        index_description = f"IVF{nlist}_Flat_nprobe_{nprobe}_{custom_name}_v2.index"
    else:
        index_description = f"{index_type}_nprobe_{nprobe}_{custom_name}_v2.index"
    
    directory, original_name = os.path.split(index_path)
    new_index_path = os.path.join(directory, index_description)
    
    os.rename(index_path, new_index_path)
    
    print(f"Original naming restored. Saved as: '{index_description}'")

if __name__ == "__main__":
    main()
