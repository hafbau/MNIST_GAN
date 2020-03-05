import torch
from torch.autograd.variable import Variable

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(discriminator, loss_fn, optimizer, real_data, fake_data):
    DATA_HEIGHT = real_data.size(0)

    # Reset optimizer
    optimizer.zero_grad()

  # Train on Real data
    prediction_real = discriminator(real_data)
    # Calculating the loss
    #   - discriminator return ones (1s) for real data and zeros (0s) for fake data
    loss_real = loss_fn(prediction_real, ones_target(DATA_HEIGHT))
    # back propagation
    loss_real.backward()

  # Train on Fake data
    prediction_fake = discriminator(fake_data)
    loss_fake = loss_fn(prediction_fake, zeros_target(DATA_HEIGHT))
    # back propagation
    loss_fake.backward()

  # Update the weights (parameters) with the back propagated gradients
    optimizer.step()

    total_loss = loss_fake + loss_real
    return total_loss, prediction_real, prediction_fake

def train_generator(discriminator, loss_fn, optimizer, generated_fake):
    DATA_HEIGHT = generated_fake.size(0)
    optimizer.zero_grad()

    # How close to real data is this generated fake,
    # as determined by the discriminator. That is, how 
    # confused is the discriminator based off this 
    # generated fake from the generator
    prediction = discriminator(generated_fake)

    # Since real data are predicted as ones (1s) by 
    # the discriminator; our loss is calculated as such
    loss = loss_fn(prediction, ones_target(DATA_HEIGHT))
    loss.backward()

    # Update the weights (parameters) with the back propagated gradients
    optimizer.step()

    return loss