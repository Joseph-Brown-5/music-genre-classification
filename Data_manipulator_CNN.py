import json
import os
import math
import librosa

DATASET_PATH = "Dataset/genres_original"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {"mapping": [], "labels": [], "mfcc": []}
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # process all audio files in genre sub-dir
            for f in filenames:
                # Ensure file is an audio file before processing
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    file_path = os.path.join(dirpath, f)
                    try:
                        signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
                        continue  # Move on to the next file

                    # process all segments of audio file
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment
                        mfcc = librosa.feature.mfcc(
                            y=signal[start:finish],
                            sr=SAMPLE_RATE,
                            n_mfcc=num_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length,
                        ).T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print(f"{file_path}, segment:{d+1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
