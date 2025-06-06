import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import tempfile
import concurrent.futures

class DeepfakeAudioDetector:
    def __init__(self, model_type='lstm', sample_rate=16000, max_workers=4):
        self.sample_rate = sample_rate
        self.max_workers = max_workers
        self.model_type = model_type.lower()
        self.model = self._build_model()

    def _build_model(self):
        """Build LSTM or XceptionNet model"""
        if self.model_type == 'lstm':
            model = models.Sequential([
                layers.Input(shape=(100, 1)),
                layers.LSTM(128, return_sequences=True),
                layers.LSTM(64),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'xception':
            model = tf.keras.applications.Xception(
                weights=None,
                input_shape=(128, 128, 3),
                classes=1,
                classifier_activation='sigmoid'
            )
        else:
            raise ValueError("Unsupported model type. Options: 'lstm' or 'xception'")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                video = VideoFileClip(video_path)
                if video.audio is None:
                    raise ValueError(f"No audio in video: {video_path}")
                video.audio.write_audiofile(tmpfile.name, verbose=False, logger=None)
                return tmpfile.name
        except Exception as e:
            raise RuntimeError(f"Video audio extraction error: {e}")

    def preprocess_audio(self, audio_path):
        """Convert audio to model input format"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        if self.model_type == 'lstm':
            padded = librosa.util.fix_length(y, size=16000 * 2)  # 2-second signal
            return padded.reshape(-1, 1)
        
        elif self.model_type == 'xception':
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S = librosa.power_to_db(S, ref=np.max)
            S = np.repeat(S[..., np.newaxis], 3, -1)  # RGB channels
            return tf.image.resize(S, (128, 128)).numpy()

    def train(self, real_paths, fake_paths, epochs=10, batch_size=8):
        """Train the model"""
        X, y = [], []
        for path in real_paths + fake_paths:
            try:
                file_type = self._detect_file_type(path)
                print(f"[INFO] Training: {file_type} - {os.path.basename(path)}")
                
                if path.endswith(('.mp4', '.avi', '.mov')):
                    temp_audio = self.extract_audio_from_video(path)
                    features = self.preprocess_audio(temp_audio)
                    os.unlink(temp_audio)
                else:
                    features = self.preprocess_audio(path)
                
                X.append(features)
                y.append(0 if path in real_paths else 1)
            
            except Exception as e:
                print(f"[WARNING] Skipped training file: {path} - {e}")

        if len(set(y)) < 2:
            raise ValueError("Training data insufficient (only real or fake samples found)")

        X = np.array(X)
        y = np.array(y)

        print(f"[INFO] Training started ({self.model_type.upper()})...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, file_path):
        """Predict single file"""
        try:
            file_type = self._detect_file_type(file_path)
            print(f"[INFO] Analyzing: {file_type} - {os.path.basename(file_path)}")
            
            if file_path.endswith(('.mp4', '.avi', '.mov')):
                temp_audio = self.extract_audio_from_video(file_path)
                features = self.preprocess_audio(temp_audio)
                os.unlink(temp_audio)
            else:
                features = self.preprocess_audio(file_path)
            
            confidence = self.model.predict(features[np.newaxis, ...])[0][0]
            return {
                "file": os.path.basename(file_path),
                "type": file_type,
                "label": "fake" if confidence > 0.5 else "real",
                "confidence": float(confidence)
            }
        
        except Exception as e:
            return {
                "file": os.path.basename(file_path),
                "type": "unknown",
                "label": "error",
                "detail": str(e)
            }

    def batch_predict(self, file_paths):
        """Batch prediction with parallel processing"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.predict, path): path for path in file_paths}
            for future in concurrent.futures.as_completed(future_to_file):
                results.append(future.result())
        return results

    def _detect_file_type(self, file_path):
        """Detect file type: audio-only, video-only, audio+video, silent video"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.mp4', '.avi', '.mov']:
            try:
                video = VideoFileClip(file_path)
                has_audio = video.audio is not None
                video.close()
                return "audio+video" if has_audio else "silent video"
            except Exception:
                return "corrupted video"
        elif ext in ['.wav', '.mp3']:
            return "audio-only"
        else:
            return "unsupported format"