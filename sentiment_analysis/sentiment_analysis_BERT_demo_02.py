import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from sklearn.metrics import confusion_matrix

# Import some Tweets from Barack Obama
df = pd.read_csv("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv")
df.head(3)

# Feature extraction using Hugging Face's BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Check the column names in the dataframe
print(df.columns)

# Assuming the correct column for the tweet text is 'content'
X = df["tweet"].values  # Use the correct column name here
y = df['hate_speech'].values  # This assumes 'harmful' is the correct label column

# Feature extraction using Hugging Face's BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sequences = [sequence for sequence in X]
model_inputs = tokenizer(sequences, padding=True, return_tensors='tf')

# Model training using TensorFlow
dataset = tf.data.Dataset.from_tensor_slices((model_inputs['input_ids'], y))

# Split dataset before shuffling to prevent data leakage
portion_for_test_percent = 0.1
portion_for_validation_percent = 0.2
portion_for_train_percent = 1 - portion_for_test_percent - portion_for_validation_percent

dataset_size = len(y)
train_size = int(dataset_size * portion_for_train_percent)
val_size = int(dataset_size * portion_for_validation_percent)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size).take(test_size)

# Shuffle datasets after splitting to avoid leakage
train_dataset = train_dataset.shuffle(160000).batch(16).prefetch(8)
val_dataset = val_dataset.batch(16).prefetch(8)
test_dataset = test_dataset.batch(16).prefetch(8)

# Model definition
model = Sequential(name="text-classifier")
model.add(Embedding(len(tokenizer.get_vocab()), 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training pipeline
history = model.fit(train_dataset, epochs=3, validation_data=val_dataset)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
test_labels = []
test_preds = []

for batch in test_dataset:
    x_true, y_true = batch
    y_pred = model.predict(x_true)
    test_labels.extend(y_true.numpy())
    test_preds.extend(y_pred)

test_labels = np.array(test_labels)
test_preds = np.array(test_preds)

# Calculate precision, recall, and accuracy
pre = Precision()
rec = Recall()
acc = Accuracy()

pre.update_state(test_labels, test_preds)
rec.update_state(test_labels, test_preds)
acc.update_state(test_labels, test_preds)

print(f"Precision: {pre.result().numpy()}")
print(f"Recall: {rec.result().numpy()}")
print(f"Accuracy: {acc.result().numpy()}")

# Confusion Matrix
cm = confusion_matrix(test_labels, (test_preds > 0.5))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Harmful', 'Harmful'], yticklabels=['Not Harmful', 'Harmful'])
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

"""
Index(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither',
       'class', 'tweet'],
      dtype='object')
Epoch 1/3
1085/1085 ━━━━━━━━━━━━━━━━━━━━ 361s 329ms/step - accuracy: 0.7660 - loss: 40.9052 - val_accuracy: 0.8265 - val_loss: -119.4103
Epoch 2/3
1085/1085 ━━━━━━━━━━━━━━━━━━━━ 394s 340ms/step - accuracy: 0.7715 - loss: -995.4325 - val_accuracy: 0.8134 - val_loss: -5778.8931
Epoch 3/3
1085/1085 ━━━━━━━━━━━━━━━━━━━━ 361s 333ms/step - accuracy: 0.6694 - loss: -59730.1133 - val_accuracy: 0.8297 - val_loss: -149872.4531
"""