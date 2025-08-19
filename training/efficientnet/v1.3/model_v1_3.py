import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math

# Desativa otimizações que podem causar problemas em algumas configurações
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = (380, 380) 
BATCH_SIZE = 16
CHANNELS = 3
DATA_DIR = './data'

# Parâmetros de Treinamento
EPOCHS_PRE_TRAIN = 20
EPOCHS_FINE_TUNE = 30
TOTAL_EPOCHS = EPOCHS_PRE_TRAIN + EPOCHS_FINE_TUNE

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

def one_hot(image, label, num_classes):
    """Converte labels para one-hot encoding."""
    return image, tf.one_hot(label, num_classes)

def mixup(image, label, alpha=MIXUP_ALPHA):
    """Aplica a técnica de aumento de dados MixUp."""
    # Amostra uma imagem e label aleatórios do mesmo batch
    batch_size = tf.shape(image)[0]
    idx = tf.random.shuffle(tf.range(batch_size))
    
    # Mistura as imagens
    lam = tf.random.uniform(shape=[], minval=0.0, maxval=alpha)
    mixed_image = (1 - lam) * image + lam * tf.gather(image, idx)
    mixed_label = (1 - lam) * label + lam * tf.gather(label, idx)
    
    return mixed_image, mixed_label

def load_and_prepare_dataset(data_dir, img_size, batch_size):
    """Carrega, pré-processa e otimiza os datasets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names) 

    # Pré-processamento
    def preprocess_train(image, label):
        image = applications.efficientnet_v2.preprocess_input(image)
        return one_hot(image, label, num_classes)

    def preprocess_val(image, label):
        image = applications.efficientnet_v2.preprocess_input(image)
        return one_hot(image, label, num_classes)

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

    # Aplica MixUp apenas no dataset de treino
    train_ds = train_ds.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)

    # Otimiza a performance
    train_ds = train_ds.cache().shuffle(1024).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_classes

def get_data_augmentation():
    """Define as camadas de aumento de dados"""
    return models.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.3),
    ], name='data_augmentation')

def build_model(num_classes, img_size, channels):
    """Define o modelo"""
    base_model = applications.EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, channels),
        pooling='avg'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(*img_size, channels))
    x = get_data_augmentation()(inputs)
    x = base_model(x, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

class WarmupCosineDecay(keras.callbacks.Callback):
    """Callback para um cronograma de Cosseno Decaído com Aquecimento."""
    def __init__(self, total_steps, warmup_steps, initial_lr, final_lr):
        super(WarmupCosineDecay, self).__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.global_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.global_step += 1
        
        if self.global_step < self.warmup_steps:
            lr = self.initial_lr * (self.global_step / self.warmup_steps)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay
        
        self.model.optimizer.learning_rate.assign(lr)

def plot_history(history, title_prefix):
    """Plota as métricas de acurácia e loss do treinamento."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia do Treinamento')
    plt.plot(epochs_range, val_acc, label='Acurácia da Validação')
    
    # Valor final anotado
    plt.text(len(acc) - 1, acc[-1], f'{acc[-1]:.4f}', ha='right', va='bottom', fontsize=9, color='blue')
    plt.text(len(val_acc) - 1, val_acc[-1], f'{val_acc[-1]:.4f}', ha='right', va='bottom', fontsize=9, color='orange')
    
    plt.legend(loc='lower right')
    plt.title(f'{title_prefix} - Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Loss do Treinamento')
    plt.plot(epochs_range, val_loss, label='Loss da Validação')
    
    # Valor final anotado
    plt.text(len(loss) - 1, loss[-1], f'{loss[-1]:.4f}', ha='right', va='top', fontsize=9, color='blue')
    plt.text(len(val_loss) - 1, val_loss[-1], f'{val_loss[-1]:.4f}', ha='right', va='top', fontsize=9, color='orange')
    
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix} - Loss (Perda)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    
    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def evaluate_model(model, val_ds, class_names):
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.show()
    
def combine_histories(history1, history2):
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history[key]
        
    return combined

if __name__ == '__main__':
    # Carrega e prepara os dados
    train_ds, val_ds, class_names, num_classes = load_and_prepare_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    
    # Constrói o modelo
    model, base_model = build_model(num_classes, IMG_SIZE, CHANNELS)
    
    # Calcula os passos para o LR Scheduler
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    warmup_steps = steps_per_epoch * 5 # 5 épocas de aquecimento
    
    lr_scheduler_cb = WarmupCosineDecay(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        initial_lr=1e-3,
        final_lr=1e-6
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    
    history_pre = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PRE_TRAIN,
        callbacks=[lr_scheduler_cb],
    )
    
    # FINE-TUNING 
    
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 60
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_pre.epoch[-1] + 1, # Continua de onde parou
        callbacks=[
            lr_scheduler_cb, # Continua usando o mesmo scheduler
            early_stopping_cb,
            keras.callbacks.ModelCheckpoint(
                './models/corn_disease_classifier.keras', 
                monitor='val_accuracy', 
                save_best_only=True,
                mode='max' # Salva o modelo com a maior acurácia de validação
            )
        ],
    )
    
    evaluate_model(model, val_ds, class_names)

    combined_history = combine_histories(history_pre, history_fine)
    plot_history(tf.keras.callbacks.History(combined_history), 'Treinamento Completo')