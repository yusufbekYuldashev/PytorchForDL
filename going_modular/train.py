
import os
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', '-train_dir', type=str, help='Training data directory', default='data/pizza_steak_sushi/train')
parser.add_argument('--test_dir', '-test_dir', type=str, help='Testing data directory', default='data/pizza_steak_sushi/test')
parser.add_argument('-learning_rate', '--learning_rate', type=float, help='Learning rate for the model', default=0.001)
parser.add_argument('-batch_size', '--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('-num_epochs', '--num_epochs', type=int, help='Number of epochs', default=10)
parser.add_argument('-hidden_units', '--hidden_units', type=int, help='Number of hidden units for the hidden layer', default=10)

args = parser.parse_args()

LEARNING_RATE = args.learning_rate 
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
HIDDEN_UNITS = args.hidden_units

train_dir = args.train_dir
test_dir = args.test_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir, test_dir, transform, BATCH_SIZE)

model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start = timer()
print('Starting training')
engine.train(model, train_dataloader, test_dataloader, optimizer, device, loss_fn, NUM_EPOCHS)
end = timer()
print('Saving the model')
utils.save_model(model, target_dir='models', model_name='05_going_modular_tinyvgg.pth')
