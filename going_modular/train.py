"""
Contains code to train the model
"""
import os
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

import argparse

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir, test_dir, transform, BATCH_SIZE)

model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parmeters(), lr=LEARNING_RATE)

start = timer()
engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=NUM_EPOCHS, device=device)
end = timer()

utils.save_model(model, target_dir='models', model_name='05_going_modular_tinyvgg.pth')
