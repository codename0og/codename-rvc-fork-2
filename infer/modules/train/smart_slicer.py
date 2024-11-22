import librosa
import numpy as np
import scipy.signal as signal

class SmartSlicer:
    def __init__(self, sr, speech_threshold=-65, min_speech_duration=0.36, min_pause_duration=0.048, chunk_duration=3.0):
        self.sr = sr
        self.speech_threshold = speech_threshold  # dB threshold for detecting speech or singing
        self.min_speech_duration = min_speech_duration  # Minimum duration to consider speech
        self.min_pause_duration = min_pause_duration  # Minimum duration for silence/glottal stop to consider as a cut point
        self.chunk_duration = chunk_duration  # Desired chunk duration (in seconds)
        self.energy_threshold = 0.01  # Threshold for low-energy (non-speech) segments, adjustable based on your input data

    def energy(self, signal):
        """ Compute the energy of the signal. """
        return np.sum(signal ** 2) / len(signal)

    def detect_speech(self, audio):
        """ Detect regions of speech based on energy. """
        frame_length = 1024  # Length of each frame
        hop_length = 512  # Hop length for moving window

        # Compute energy of each frame
        energy_vals = []
        for start in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start:start + frame_length]
            energy_vals.append(self.energy(frame))
        
        # Detect speech segments based on energy
        speech_segments = []
        is_speech = False
        for idx, energy in enumerate(energy_vals):
            if energy > self.energy_threshold:  # Speech detected
                if not is_speech:
                    start_idx = idx * hop_length
                    is_speech = True
            else:  # Pause or breathing detected
                if is_speech:
                    end_idx = idx * hop_length
                    speech_segments.append((start_idx, end_idx))
                    is_speech = False
        if is_speech:  # Append the last speech segment if it ended before the audio ends
            speech_segments.append((start_idx, len(audio)))

        return speech_segments

    def slice(self, audio):
        """ Slice the audio into chunks based on the speech and pause detection. """
        speech_segments = self.detect_speech(audio)
        chunks = []
        chunk_size = int(self.chunk_duration * self.sr)
        
        # Go through the speech segments and break them into chunks
        for start, end in speech_segments:
            segment = audio[start:end]
            segment_length = len(segment)
            
            # If the segment is longer than the desired chunk, break it into multiple chunks
            while len(segment) > chunk_size:
                chunk = segment[:chunk_size]
                chunks.append(chunk)
                segment = segment[chunk_size:]
            
            # If the remaining segment is longer than a small amount, add it as a chunk
            if len(segment) > 0:
                chunks.append(segment)

        return chunks

    def smooth_cut(self, audio, start, end):
        """ Make a smooth cut by checking the end of the slice to avoid zero-crossings. """
        # Check if the cut falls near a zero-crossing and adjust the cut point if necessary
        slice = audio[start:end]
        # Find the nearest zero crossing to make the cut smoother (avoiding abrupt transitions)
        cross_points = np.where(np.diff(np.sign(slice)))[0]
        if len(cross_points) > 0:
            cut_point = start + cross_points[0]  # First zero-crossing point
            return audio[start:cut_point]
        else:
            return slice

    def process_audio(self, audio):
        """ Main processing function to slice the audio. """
        slices = self.slice(audio)
        final_chunks = []

        # Ensure every slice is exactly 3 seconds, handle smooth cuts
        for slice in slices:
            # If slice is too short, skip it
            if len(slice) < self.sr * self.min_speech_duration:
                continue
            start = 0
            while start + self.sr * self.chunk_duration <= len(slice):
                end = start + self.sr * self.chunk_duration
                chunk = self.smooth_cut(slice, start, end)
                final_chunks.append(chunk)
                start = end

        return final_chunks

# Usage example:
sr = 16000  # Sampling rate
audio, sr = librosa.load('input.wav', sr=sr)  # Load your audio
slicer = SmartSlicer(sr)

chunks = slicer.process_audio(audio)
for i, chunk in enumerate(chunks):
    # Save each chunk (ensure it's 3 sec long or as close to it as possible)
    librosa.output.write_wav(f'chunk_{i}.wav', chunk, sr)
