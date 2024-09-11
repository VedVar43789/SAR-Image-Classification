import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2

# Speckle Filter Function (Lee Filter)
def lee_filter(image, size=7):
    img_mean = cv2.blur(image, (size, size))
    img_sqr_mean = cv2.blur(image ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = np.var(image)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (image - img_mean)
    return img_output.astype(np.uint8)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

# Combined Preprocessing Function
def preprocessing_function(image):
    image = lee_filter(image)  
    image = apply_clahe(image)  
    return image / 255.0  

# Data Augmentation and Preprocessing
train_data_gen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocessing_function,
    rotation_range=10,  
    width_shift_range=0.05,  
    height_shift_range=0.05,  
    shear_range=0.05,  
    zoom_range=0.05,  
    horizontal_flip=True,  
    vertical_flip=True,  
    fill_mode='nearest'
)

train_generator = train_data_gen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),  
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_data_gen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),  
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Custom Model on Top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze Base Model Layers
for layer in base_model.layers:
    layer.trainable = False  

# Compiling Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train Model
history = model.fit(train_generator, epochs=10,  
                    validation_data=validation_generator,
                    callbacks=callbacks)

# Plot Model Accuracy and Model Loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Evaluate Model
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Classification Report
target_names = list(validation_generator.class_indices.keys())
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=target_names))