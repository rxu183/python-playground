#Ok, so this will be a real GAN, coded by me.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import os
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
BATCH_SIZE = 3
EPOCHS = 10
NUMBER_TO_GENERATE = 3
k = 1 #Number of times to train the discriminator before training the generator.


writer = SummaryWriter() #Defines tensorboard writer.

#0. Create the Data Reading Class
class img_datset(torch.utils.data.Dataset):
    #OUr dataset requires three elements:
    def __init__(self, dataset_location, has_classes=False, has_annotations=False):
        self.has_classes = has_classes #I'm saving this in case I want ot turn this into a classifier at some point
        self.annotations = has_annotations #Similar reason to above. Currently, I just need images.
        self.data_loc = dataset_location #This is the actual data.
        self.classes = None
        if has_classes:
            self.classes = os.listdir(dataset_location) #List of classes in our dataset
            self.data = []
            for category in self.classes:
                self.data.append(os.listdir(os.path.join(dataset_location, category))) #List of files in each class
        else:
            self.data = os.listdir(dataset_location) #We're working with data directly
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        #At least for our GAN:
        image_loc = os.path.join(self.data_loc, self.data[index])
        image = cv2.imread(image_loc)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
#1. Create the generator model
class generator(torch.nn.Module):
    def __init__(self):
        #Ok, so the way this works is um we just throw more nodes at our model. 
        super().__init__()
        self.input = torch.nn.Linear(100, 16 * 16 * 64) #Create 64 16x16 panels that are the result of something.
        self.activation1 = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding="same") #oh good heavens this is complicated ot iamgine. Padding refers to the padding necessary to generat this output.
        self.activation2 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding="same")
        self.activation3 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding="same")
        self.activation4 = torch.nn.LeakyReLU()
        self.output = torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding="same")
    def forward(self, input): #I CAN INPUT A BATCH OF NOISE VECTORS
        x = self.input(input)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.activation3(x)
        x = self.conv3(x)
        x = self.activation4(x)
        x = self.output(x)
        return x
model_generator = generator()

#2. Create the discriminator model
#Last dimension is number of features.
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Conv2d(3, 16, 3, stride=2, padding="same") #3 input channels, 16 output channels, 3x3 kernel, stride 2, padding same
        self.activation1 = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=2, padding="same")
        self.activation2 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding="same")
        self.activation3 = torch.nn.LeakyReLU()
        self.flatten = torch.nn.Flatten() #Ok, nice this flattens it for us without 
        self.output = torch.nn.Linear(BATCH_SIZE, 1, bias=True) #Output dimension.
        self.activationOut = torch.nn.sigmoid()
        #I should have roughly 2916 trainable pamaeters.
    
    def forward(self, fake, real):
        x = self.input(fake)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.activation3(x)
        x = self.flatten(x)
        x = self.output(x)
        x = self.activationOut(x)

        y = self.input(real)
        y = self.activation1(y)
        y = self.conv1(y)
        y = self.activation2(y)
        y = self.conv2(y)
        y = self.activation3(y)
        y = self.flatten(y)
        y = self.output(y)
        return x, y
model_discriminator = discriminator()

#3. Create the loss functions

discriminator_loss = torch.nn.BCELoss()
generator_loss = torch.nn.BCELoss()
#4. Create the optimizers
discriminator_optimizer = torch.nn.optim.Adam(model_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator_optimizer = torch.nn.optim.Adam(model_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#5. Create the training loop
def train(disc_in, model_disciminator, model_generator, generator_loss, discriminator_loss, discriminator_optimizer, generator_optimizer):
    for epoch in range(EPOCHS):
        #This is going to suck. A lot - so we can train the discriminator a few times, and then the generator once. 
        #We need to 
        #Forward pass
        step = 0
        for index, batch in disc_in:
            #Train the discriminator, possibly multiple times per a given batch.
            for num_iterate in range(k):
                print("We are on step: ", num_iterate, " which should reflect difference in " , index)
                noise = torch.randn(BATCH_SIZE, 100, requires_grad=True) #This generates a batch of noise vectors - note that this indicates BATCH_SIZE number of vectors generated? I hope anyway.
                fakes = model_generator(noise)
                prob_fake, prob_real = model_disciminator(fakes, batch)
                #Calculate the loss
                #Need to create a tensor of ones and zeros
                zeros = torch.zeros(BATCH_SIZE, 1)
                ones = torch.ones(BATCH_SIZE, 1)
                disc_loss = discriminator_loss(prob_fake, zeros)
                disc_loss += discriminator_loss(prob_real, ones) #Do we just sum the losses?
                SummaryWriter.add_scalar(writer, "Discriminator Loss", disc_loss, step)

                disc_loss.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()
                print("We have trained the discriminator!")
            #Train the generator - once per batch.
            #First, we need to freeze weights of discriminator
            for param in discriminator.parameters():
                param.requires_grad = False
            #Now we can train the generator
            noise = torch.randn(BATCH_SIZE, 100)
            fakes = model_generator(noise)
            probabilities_fake, probabilities_real = model_disciminator(fakes, batch) #Note batch = real iamges.
            gen_loss = generator_loss(probabilities_fake, ones) #Maximize the probability the discriminator thinks that the images the generator made are real?
            SummaryWriter.add_scalar(writer,"Generator Loss", gen_loss, step)
            gen_loss.backward()
            generator_optimizer.step() #Use ADAM optimizer
            generator_optimizer.zero_grad() #Clear Gradients

#6. Create the test loop - there's no test loss, really. We just run and look at the output
def test(model_generator):
    noise = torch.randn(NUMBER_TO_GENERATE, 100)
    fakes = model_generator(noise)
    for index in range(NUMBER_TO_GENERATE):
        plt.imshow(fakes[index])
        plt.show()


        #Possibly if we need annotation support or something we can add more later:

#7. Create the main function with visualization
def main():
    gen_test = generator()
    data = img_dataset(os.path.join('.', 'dataset', 'actual'))

#8. Run the main function!
main()


