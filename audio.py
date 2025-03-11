import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "GTZAN/"

GENRE_TO_MOOD = {
    "blues": "Sad", "classical": "Calm", "country": "Happy", "disco": "Energetic",
    "hiphop": "Energetic", "jazz": "Calm", "metal": "Energetic", "pop": "Happy",
    "reggae": "Happy", "rock": "Energetic"
}

MOOD_EMOJIS = {
    "Happy": "😃", "Sad": "😢", "Calm": "😌", "Energetic": "⚡"
}

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])
data = []
for genre in tqdm(os.listdir(DATASET_PATH)):
    genre_path = os.path.join(DATASET_PATH, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            mood = GENRE_TO_MOOD[genre]
            data.append([features, mood])

df = pd.DataFrame(data, columns=["features", "mood"])
df.to_pickle("music_features.pkl")  
print("✅ Feature Extraction Complete!")

df = pd.read_pickle("music_features.pkl")
X = np.vstack(df["features"].values)
y = df["mood"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MoodClassifier(nn.Module):
    def __init__(self):
        super(MoodClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(set(y_encoded)))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MoodClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):  
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

print("✅ Model Training Complete!")

YOUTUBE_API_KEY = "your_youtube_api_key_here"

def fetch_youtube_songs(mood):
    """Fetch top YouTube playlists based on mood."""
    query = f"{mood} music playlist"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=playlist&maxResults=5&key={YOUTUBE_API_KEY}"

    try:
        response = requests.get(url).json()
        playlists = response.get("items", [])

        if not playlists:
            print(f"⚠️ No playlists found for mood: {mood}")
            return

        print(f"\n🎵 Top YouTube Playlists for Mood: {mood}")
        for playlist in playlists:
            title = playlist["snippet"]["title"]
            playlist_id = playlist["id"]["playlistId"]
            print(f"📺 {title} - https://www.youtube.com/playlist?list={playlist_id}")

    except Exception as e:
        print(f"❌ Error fetching YouTube data: {e}")

def visualize_audio(audio_path):
    """Show waveform & spectrogram visualization."""
    y, sr = librosa.load(audio_path, duration=30)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Spectrogram
    plt.subplot(2, 1, 2)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis="time", y_axis="mel")
    plt.title("Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

def predict_mood(audio_file):
    """Predict mood of a song, show visual effects & fetch related YouTube playlists."""
    features = extract_features(audio_file)
    features_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
    mood_index = model(features_tensor).argmax().item()
    predicted_mood = label_encoder.inverse_transform([mood_index])[0]

    emoji = MOOD_EMOJIS.get(predicted_mood, "🎵")
    print(f"🎶 Detected Mood: {predicted_mood} {emoji}")

    visualize_audio(audio_file)

    fetch_youtube_songs(predicted_mood)

predict_mood("test_song.mp3")
  
