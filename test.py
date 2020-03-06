import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger, mnist_data, images_to_vectors, vectors_to_images, noise
from discriminator import DiscriminatorNet
from generator import GeneratorNet
from trainers import train_discriminator, train_generator

# Hack for MNIST fetch error
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)


# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

num_test_samples = 16
test_noise = noise(num_test_samples, use_cuda)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        
        # use cuda if available
        if use_cuda:
            real_batch = real_batch.cuda()

        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        print('is CUDA', next(generator.parameters()).is_cuda)
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N, use_cuda)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(discriminator, loss, d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N, use_cuda))
        # Train G
        g_error = train_generator(discriminator, loss, g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )