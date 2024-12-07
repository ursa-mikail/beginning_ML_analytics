import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset

# Load BERT model and tokenizer
model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the sample input
inputs = tokenizer(['Hello world', 'Hi how are you'], padding=True, truncation=True, return_tensors='tf')

# Load emotion dataset
emotions = load_dataset('SetFit/emotion')

# Tokenize emotion dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

# Set BATCH_SIZE to 64
BATCH_SIZE = 64

# Function to prepare data for the model
def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

# Prepare train and test datasets
train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# Define the model for classification
class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

# Initialize the model
classifier = BERTForClassification(model, num_classes=6)

# Compile the model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model and capture training history
history = classifier.fit(
    train_dataset,
    epochs=3
)

# Evaluate the model on the test set
test_loss, test_acc = classifier.evaluate(test_dataset)

# Print test accuracy
print(f"Test accuracy: {test_acc}")

# Plot the training loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()

