import tensorflow as tf
from data_preprocessing import prepare_dataset
from model import build_resnet

# Load and preprocess the data
data_dir = "C:\\Users\\ADMIN\\Documents\\Workspace\\mri-cnn-project\\mri-cnn-project\\data\\raw"
X_train, X_test, y_train, y_test = prepare_dataset(data_dir)

# Build the model
model = build_resnet((224, 224, 1), num_classes=len(set(y_train)))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

# Adjust training parameters
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    tf.keras.callbacks.ModelCheckpoint('model_best.h5', save_best_only=True)
]

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('mri_classification_model.h5')