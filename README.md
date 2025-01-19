# Speech Emotion Recognition System

## Overview

This project implements an **Emotion Recognition System** from speech using **Deep Learning** and **Speech Processing** techniques. The goal of this project is to classify spoken sentences or phrases into different emotions like **happiness**, **anger**, **sadness**, and others based on audio features extracted from the speech. The model utilizes **Mel Frequency Cepstral Coefficients (MFCCs)** as features and trains a **Recurrent Neural Network (RNN)** or **Convolutional Neural Network (CNN)** for classification.

### Key Features:
- **Emotion Classification**: The system recognizes various emotions from speech (e.g., happiness, anger, sadness).
- **Deep Learning Models**: The system uses deep learning models such as RNNs (LSTM) or CNNs for speech emotion classification.
- **Speech Preprocessing**: Audio features like MFCC are extracted from the raw audio data.
- **Dataset**: Utilizes publicly available emotion-labeled datasets such as **RAVDESS** or **TESS**.

---

## Table of Contents

- [Project Description](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

1. **Python 3.x**: This project is developed using Python 3.x.
2. **Required Libraries**: The project requires the following libraries for deep learning and audio processing:
   - `tensorflow` or `keras`
   - `librosa` (for feature extraction)
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `pandas`
   - `soundfile`

You can install all required libraries by running:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
tensorflow==2.7.0
librosa==0.8.1
numpy==1.21.2
matplotlib==3.4.3
scikit-learn==0.24.2
pandas==1.3.3
soundfile==0.10.3.post1
```

---

## Usage

### 1. Preprocess Audio Data

First, you need to preprocess the audio files. This involves extracting features like **MFCCs** (Mel Frequency Cepstral Coefficients), which represent the audio in a form suitable for deep learning models.

```python
import librosa
import numpy as np

def extract_mfcc_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Take the mean of the MFCC coefficients across the time axis
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean
```

### 2. Load Pre-trained Model

If you already have a trained model, you can load it using the following code:

```python
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('path_to_model/emotion_recognition_model.h5')
```

### 3. Predict Emotion from Speech

After preprocessing the audio file and extracting the MFCC features, you can pass them to the trained model to predict the emotion.

```python
def predict_emotion(audio_file, model):
    # Extract features from the audio
    features = extract_mfcc_features(audio_file)
    
    # Reshape features to match model input shape (e.g., [1, 13])
    features = features.reshape(1, -1)
    
    # Predict emotion
    prediction = model.predict(features)
    predicted_emotion = np.argmax(prediction)
    
    return predicted_emotion
```

### 4. Mapping Predictions to Emotions

If your model predicts a number, you can map the prediction to the corresponding emotion:

```python
emotion_dict = {0: 'Anger', 1: 'Happiness', 2: 'Sadness', 3: 'Fear'}

# Example usage
audio_file = 'path_to_audio.wav'
predicted_emotion = predict_emotion(audio_file, model)
print(f"Predicted Emotion: {emotion_dict[predicted_emotion]}")
```

---

## Model Training

### 1. Dataset

The system can be trained using publicly available datasets like **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) or **TESS** (Toronto Emotional Speech Set), which contain audio recordings of emotional speech with labeled emotions.

### 2. Preprocessing the Data

To prepare the data for training, you need to extract the features (MFCCs) from all audio files:

```python
import os
import librosa

def preprocess_data(dataset_path):
    features = []
    labels = []
    
    # Loop through all audio files in the dataset
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            emotion = file.split('_')[2]  # Extract emotion from the file name (e.g., "happy", "angry")
            mfcc = extract_mfcc_features(file_path)
            features.append(mfcc)
            labels.append(emotion)
    
    return np.array(features), np.array(labels)
```

### 3. Building the Model

We can build a deep learning model using **LSTM (Long Short-Term Memory)** or **CNN** to classify emotions from the extracted MFCC features.

Example using a simple **LSTM** model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # Assuming 4 emotions: Anger, Happiness, Sadness, Fear
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

### 4. Train the Model

```python
# Prepare the dataset
X_train, y_train = preprocess_data('path_to_dataset')

# Build and train the model
model = build_model(input_shape=(X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('emotion_recognition_model.h5')
```

---

## Evaluation

After training the model, you can evaluate its performance using a test dataset:

```python
# Evaluate the model on test data
X_test, y_test = preprocess_data('path_to_test_dataset')
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
```

---

## Technologies

This project is built using the following technologies:

- **Python 3.x**
- **TensorFlow / Keras**: For building and training deep learning models
- **Librosa**: For audio feature extraction (MFCC)
- **NumPy**: For numerical operations and data manipulation
- **Matplotlib**: For visualizing model performance (e.g., accuracy, loss)
- **Scikit-learn**: For additional utilities (e.g., accuracy calculation)
- **Soundfile**: For reading and processing audio files

---

## Contributing

We welcome contributions to improve this project. You can contribute by:

- Improving the model architecture (e.g., using advanced models like CNN, GRU, or Transformer-based models).
- Adding new functionalities, such as emotion recognition from real-time speech.
- Improving the feature extraction process to capture more nuances of emotion in speech.

To contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a new pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Customizing the Template:

- **Dataset**: If you are using a different dataset, update the dataset section and preprocessing logic accordingly.
- **Model**: If you choose a different architecture (e.g., CNN or Transformer-based models), modify the model-building section to reflect that.
- **Real-time Recognition**: If you add functionality for real-time emotion recognition, add relevant instructions in the **Usage** section.

