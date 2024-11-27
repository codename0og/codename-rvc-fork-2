import os
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tqdm
import multiprocessing

def detect_sample_rate(input_path):
    # Load the audio using librosa to detect sample rate
    y, sr = librosa.load(input_path, sr=None)  # sr=None to preserve original sample rate
    return sr

def probe_bit_depth(input_path):
    # Use soundfile to read the file and get the bit depth
    with sf.SoundFile(input_path) as file:
        bit_depth = file.subtype  # Get the subtype (e.g., PCM_16, PCM_24)
    return bit_depth

def choose_slice_length():
    print("Choose the slice length:")
    print("1 = 3 seconds")
    print("2 = 3.7 seconds")
    
    choice = input("Enter the number for the slice length (1 or 2): ")
    
    slice_lengths = {
        "1": 3000,  # 3 seconds in milliseconds
        "2": 3700   # 3.7 seconds in milliseconds
    }
    
    return slice_lengths.get(choice, 3000)

def slice_audio_segment(y, sr, start_ms, end_ms, output_folder, index):
    # Convert milliseconds to samples
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)
    
    # Extract slice
    slice_audio = y[start_sample:end_sample]
    
    # Export the slice as a new .wav file using soundfile.write
    slice_filename = os.path.join(output_folder, f"slice_{index}.wav")
    sf.write(slice_filename, slice_audio, sr)
    return slice_filename

def slice_audio(input_path, output_folder, slice_length_ms):
    # Load the audio using librosa for better handling of sample rate and bit depth
    y, sr = librosa.load(input_path, sr=None)  # sr=None keeps original sample rate
    
    # Get audio duration in milliseconds
    duration_ms = len(y) / sr * 1000  # Convert to milliseconds
    num_slices = int(duration_ms // slice_length_ms)  # total number of slices
    
    discarded_length = 0  # Track discarded length
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare tasks for the thread pool
    tasks = []
    for i in range(num_slices):
        start_ms = i * slice_length_ms
        end_ms = start_ms + slice_length_ms
        tasks.append((y, sr, start_ms, end_ms, output_folder, i))
    
    # Handle multithreading with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 2)) as executor:
        futures = [executor.submit(slice_audio_segment, *task) for task in tasks]
        
        # Using tqdm for a progress bar
        for future in tqdm.tqdm(futures, desc="Slicing Audio", unit="slice"):
            future.result()  # Wait for all tasks to complete
    
    # Handle remaining audio after full slices
    remaining_ms = duration_ms - (num_slices * slice_length_ms)
    if remaining_ms > 0:
        if remaining_ms < slice_length_ms:
            discarded_length += remaining_ms  # Discard remaining part
            print(f"The last segment was discarded due to insufficient length.")
        else:
            # Save the last segment if it fits the required length
            start_ms = num_slices * slice_length_ms
            end_ms = start_ms + remaining_ms
            last_slice_audio = y[int(start_ms*sr/1000):int(end_ms*sr/1000)]  # Convert ms to samples
            last_slice_filename = os.path.join(output_folder, f"slice_{num_slices}.wav")
            sf.write(last_slice_filename, last_slice_audio, sr)
            print(f"Exported {last_slice_filename}")
    
    print(f"Total discarded length: {discarded_length / 1000} seconds.")

def main():
    # Get input file path
    input_path = input("Enter the path to the .wav sample: ")
    if not os.path.exists(input_path):
        print("File not found.")
        return
    
    # Auto-detect sample rate using librosa
    sample_rate = detect_sample_rate(input_path)
    print(f"Detected sample rate: {sample_rate} Hz")
    
    # Probe bit depth using soundfile
    bit_depth = probe_bit_depth(input_path)
    print(f"Detected bit depth: {bit_depth}")
    
    # Choose slice length
    slice_length_ms = choose_slice_length()
    
    # Get output folder
    output_folder = input("Enter the output folder for slices: ")
    
    # Slice the audio into segments
    slice_audio(input_path, output_folder, slice_length_ms)
    
    # Prompt user to exit
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
