import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x
    
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    
    return x

def build_resnet(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the initial layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Custom layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

# Usage
if __name__ == "__main__":
    model = build_resnet((224, 224, 1), num_classes=2)
    model.summary()