import streamlit as st # type: ignore
import librosa # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import soundfile as sf # type: ignore
from pydub import AudioSegment # type: ignore
import io

# Load the pre-trained model
model = load_model('saved_model.hdf5')

# Set page configuration
st.set_page_config(page_title="SoundSense", page_icon="🔊")

# Custom CSS for monospace font family
st.markdown(
    """
    <style>
    body {
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to extract features
def feature_extractor(filename):
    # Load the audio file
    audio, sample_rate = librosa.load(filename, sr=None)
    
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Compute the mean of the MFCCs
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    return mfccs_scaled_features

# Function to convert audio file to .wav
def convert_to_wav(file):
    audio = AudioSegment.from_file(file)
    buffer = io.BytesIO()
    audio.export(buffer, format='wav')
    buffer.seek(0)
    return buffer

# Streamlit interface
st.title("Audio Classification App")
st.write("Upload an audio file and the model will predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "ogg", "flac"])

if uploaded_file is not None:
    # Convert to .wav if necessary
    if uploaded_file.name.split('.')[-1] != 'wav':
        uploaded_file = convert_to_wav(uploaded_file)

    # Save the uploaded file to disk
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    mfccs_scaled_features = feature_extractor("temp.wav")

    # Reshape the features to match the input shape expected by the model
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Predict the class probabilities
    prediction = model.predict(mfccs_scaled_features)

    # Get the class with the highest probability
    predicted_label = np.argmax(prediction, axis=1)

    # Assuming you have a LabelEncoder fitted with your class names
    labelencoder = LabelEncoder()
    # Replace 'class_names' with your actual class names used for training
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
                   'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    labelencoder.fit(class_names)

    # Get the predicted class name
    prediction_class = labelencoder.inverse_transform(predicted_label)
    st.write(f"The predicted class is: {prediction_class[0]}")
