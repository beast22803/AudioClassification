import os
import librosa
import math
import json
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "genres_original"
JSON_PATH = "Data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # Dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Ensure we're not at the root level
        if dirpath != dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Process files for specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                # Only process audio files
                if f.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    try:
                        # Load audio file
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                        # Process segments extracting mfcc and storing data
                        for s in range(num_segments):
                            start_sample = num_samples_per_segment * s
                            finish_sample = start_sample + num_samples_per_segment

                            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                            mfcc = mfcc.T

                            # Store mfcc for segment if it has the expected length
                            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(i - 1)
                                print("{}, segment:{}".format(file_path, s))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                else:
                    print(f"Skipping non-audio file: {file_path}")
                        
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
