import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class SalmonDiseaseClassifier:
    def __init__(self, data_dir):
        # Dapatkan path absolut direktori project
        self.data_dir = data_dir
        self.input_shape = (224, 224, 3)
        self.batch_size = 32
        self.epochs = 20
        self.classes = ['FreshFish', 'InfectedFish']
    
    def load_images(self):
        images = []
        labels = []
        
        print(f"Loading images from: {self.data_dir}")
        
        for label_idx, label in enumerate(self.classes):
            path = os.path.join(self.data_dir, label)
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")
            
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                try:
                    # Baca gambar dengan OpenCV
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize gambar
                    img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                    
                    # Normalisasi
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def create_model(self):
        model = Sequential([
            # Layer konvolusi pertama
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Layer konvolusi kedua
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Layer konvolusi ketiga
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten layer
            Flatten(),
            
            # Dense layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Kompilasi model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        # Load gambar
        images, labels = self.load_images()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Buat model
        model = self.create_model()
        
        # Callback untuk early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Latih model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            callbacks=[early_stopping]
        )
        
        # Evaluasi model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        
        # Prediksi pada test set
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
        # Buat confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, 
                    yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Simpan model
        model.save('salmon_disease_model.keras')  # Update this line
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        # Cetak laporan klasifikasi
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        return model, history

# Jalankan training
if __name__ == '__main__':
    try:
        # Sesuaikan path dengan struktur folder Anda
        data_dir = 'data\sample_images'
        classifier = SalmonDiseaseClassifier(data_dir)
        model, history = classifier.train()
    except KeyboardInterrupt:
        print("\n\nTraining dihentikan oleh pengguna.")
        
        # Simpan model meskipun training dihentikan
        if 'model' in locals():
            try:
                model.save('salmon_disease_model.keras')
                print("Model berhasil disimpan sebelum training dihentikan.")
            except Exception as e:
                print(f"Gagal menyimpan model: {e}")