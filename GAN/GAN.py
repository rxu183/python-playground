#Ok, so this will be a real GAN, coded by me.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import os
import torchvision.transforms as transform
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
BATCH_SIZE = 4
EPOCHS = 300
NUMBER_TO_GENERATE = 5
MAX_TRAIN = 20
PATH = os.path.join('.', 'formatted_dataset', 'EBAY')
k = 1 #Number of times to train the discriminator before training the generator.
ITERATE_GENERATOR = range(1) #Number of times to train the generator before training the discriminator.

writer = SummaryWriter() #Defines tensorboard writer.


def transform_image(image):
    #This transforms the image to a tensor
    transformator = transform.Compose([transform.ToTensor()])
    return transformator(image)

#0. Create the Dataset sublcass
class img_datset(torch.utils.data.Dataset):
    #OUr dataset requires three elements:
    def __init__(self, dataset_location, has_classes=False, has_annotations=False):
        self.has_classes = has_classes #I'm saving this in case I want ot turn this into a classifier at some point
        self.annotations = has_annotations #Similar reason to above. Currently, I just need images.
        self.data_loc = dataset_location #This is the actual data.
        self.classes = None
        trained = 0
        if has_classes:
            self.classes = os.listdir(dataset_location) #List of classes in our dataset
            self.data = []
            for category in self.classes:
                self.data.append(os.listdir(os.path.join(dataset_location, category))) #List of files in each class
                trained += 1
                if trained > MAX_TRAIN:
                    break
        else:
            self.data = os.listdir(dataset_location) #We're working with data directly
            if(len(self.data) > MAX_TRAIN):
                self.data = self.data[:MAX_TRAIN]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        #At least for our GAN:
        image_loc = os.path.join(self.data_loc, self.data[index])
        image = cv2.imread(image_loc)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformator = transform.Compose([transform.ToTensor()]) #This defines an image - torch
        #print(transformator(image))
        return transformator(image)
#1. Create the generator model
class generator(torch.nn.Module):
    def __init__(self):
        #Ok, so the way this works is um we just throw more nodes at our model. 
        super().__init__()
        self.input = torch.nn.Linear(100, 512 * 4 * 4) #Create 256 4x4 pixels that are the result 
        self.activation1 = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True) #Layer 1
        self.activation2 = torch.nn.LeakyReLU()
        #self.dropout1 = torch.nn.Dropout2d(p=0.1)
        self.norm = torch.nn.BatchNorm2d(512)
        #self.up_sample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True) # Layer 2
        self.activation3 = torch.nn.LeakyReLU()
        #self.dropout2 = torch.nn.Dropout2d(p=0.1)
        self.norm1 = torch.nn.BatchNorm2d(256)
        #self.up_sample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True) #Layer 3
        self.activation4 = torch.nn.LeakyReLU()
        #self.dropout3 = torch.nn.Dropout2d(p=0.1)
        self.norm2 = torch.nn.BatchNorm2d(128)
        #self.activation5 = torch.nn.LeakyReLU()
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True) #Layer 4
        self.activation5 = torch.nn.LeakyReLU()
        self.norm3 = torch.nn.BatchNorm2d(64)
        #self.fin = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = torch.nn.Tanh()

        
    def forward(self, input): #I CAN INPUT A BATCH OF NOISE VECTORS
        x = self.input(input)
        #x = input
        #torch.nn.functional.normalize(x, p=2.0, dim = 1)
        #x = self.activation1(x)
        x = torch.reshape(x, (BATCH_SIZE, 512, 4, 4)) #This should reshape the input 
        #x = self.up_sample(x)
        x = self.conv1(x)
        x = self.activation2(x)
        
        #x = self.up_sample1(x)
        #x = self.dropout1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.activation3(x)
        
        #x = self.dropout2(x)
        x = self.norm1(x)
        #x = self.up_sample3(x)
        x = self.conv3(x)
        x = self.activation4(x)
        #x = self.dropout3(x)
        x = self.norm2(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.activation5(x)
        #print(x.shape)
        x = self.norm3(x)
        #x = self.fin(x)
        x = self.conv5(x)
        x = self.output(x)
        #x = (x + 1)/2
        #print(x.shape)
        #print(x)
        return x
model_generator = generator()

#2. Create the discriminator model
#Last dimension is number of features.
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) #3 input channels, 16 output channels, 3x3 kernel, stride 2, padding same
        self.activation1 = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.activation2 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.activation3 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.activation4 = torch.nn.LeakyReLU()
        self.flatten = torch.nn.Flatten(1) #Ok, nice this flattens it for us with regard to BATCH_size 
        self.dropout1 = torch.nn.Dropout(p=0.4)
        self.output = torch.nn.Linear(in_features=8192, out_features=1, bias=True) #Output dimension.
        self.activationOut = torch.nn.Sigmoid()
        #I should have roughly 2916 trainable pamaeters. Apparently i have roughly 70000? What?
    
    def forward(self, fake, real):
        x = self.input(fake)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.activation3(x)
        x = self.conv3(x)
        x = self.activation4(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.output(x)
        x = self.activationOut(x)

        y = self.input(real)
        y = self.activation1(y)
        y = self.conv1(y)
        y = self.activation2(y)
        y = self.conv2(y)
        y = self.activation3(y)
        y = self.conv3(y)
        y = self.activation4(y)
        y = self.flatten(y)
        y = self.dropout1(y)
        y = self.output(y)
        y = self.activationOut(y)
        #print(x, y)
        return x, y
model_discriminator = discriminator()

#3. Create the loss functions
discriminator_loss = torch.nn.BCELoss() #In our case, both the discriminator and generator use the same loss function.
generator_loss = torch.nn.BCELoss() #This is the loss function for the generator. We want to minimize the difference between the real and fake images.

#4. Create the optimizers
discriminator_optimizer = torch.optim.Adam(model_discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
generator_optimizer = torch.optim.Adam(model_generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
#5. Create the training loop
def train(disc_in, model_disciminator, model_generator, discriminator_loss, generator_loss, discriminator_optimizer, generator_optimizer, step=0):
    #print(EPOCHS)
    for epoch in range(EPOCHS):
        for index, batch in enumerate(disc_in):
            #batch = batch[0] #We need this when droping the class that CIFAR comes with:
            step += 1
            zeros = torch.zeros(BATCH_SIZE, 1, requires_grad=False)
            ones = torch.ones(BATCH_SIZE, 1, requires_grad=False)
            noise = torch.randn(BATCH_SIZE, 100, requires_grad=False)
            for num_iterate in range(k):
                for param in model_discriminator.parameters(): #Activate parameters
                    param.requires_grad = True
                print("We are on step: ", step, " which should reflect difference in " , index)
                fakes = model_generator(noise)
                prob_fake, prob_real = model_disciminator(fakes, batch)
                disc_loss = (discriminator_loss(prob_fake, zeros) + discriminator_loss(prob_real, ones)) #Do we just sum the losses?
                print("Disc_loss applied on Zeros", disc_loss)
                print( "Fake Probability: ", prob_fake)
                disc_loss.backward()
                weight_update_magnitude = 0.0
                for param in model_disciminator.parameters():
                    weight_update = param.grad.data  # Gradient of the parameter
                    weight_update_magnitude += weight_update.abs().sum().item()
                SummaryWriter.add_scalar(writer, "Discriminator Loss", disc_loss, step)
                SummaryWriter.add_scalar(writer, "Discriminator Weight Update Magnitude", weight_update_magnitude, step)
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()
            for i in ITERATE_GENERATOR:
                for param in model_discriminator.parameters():
                    param.requires_grad = False
                fakes = model_generator(noise)
                probabilities_fake, probabilities_real = model_disciminator(fakes, batch) #Note batch = real iamges.
                gen_loss = generator_loss(probabilities_fake, ones) #Maximize the probability the discriminator thinks that the images the generator made are real?
                print("Generator loss Applied on Ones:", gen_loss)
                print( "Probability fake: ", probabilities_fake)
                weight_update_magnitude = 0.0
                gen_loss.backward()
                for param in model_generator.parameters():
                    weight_update = param.grad.data  # Gradient of the parameter
                    weight_update_magnitude += abs(weight_update.abs().sum().item())
                SummaryWriter.add_scalar(writer, "Generator Weight Update Magnitude", weight_update_magnitude, step)
                SummaryWriter.add_scalar(writer,"Generator Loss", gen_loss, step)
                #gen_loss.backward()
                generator_optimizer.step() #Use ADAM optimizer
                generator_optimizer.zero_grad()
            #model_state_dict = model_generator.state_dict()
            #layer_weights = model_state_dict['conv1.weight']

#6. Create the test loop - there's no test loss, really. We just run and look at the output
def test(model_generator):
    for index in range(NUMBER_TO_GENERATE):
        noise = torch.randn(BATCH_SIZE, 100)
        fakes = model_generator(noise)
        fakes = (fakes + 1)/2
        for item in fakes:
            figure = plt.figure("Generated Sample")
            plt.axis("off")
            plt.imshow(np.transpose(item.detach().numpy(), (1,2,0)))
            plt.show()
        #Possibly if we need annotation support or something we can add more later:

#7. Create the main function
def main():
    data = img_datset(PATH) #PATH
    #cifar = datasets.CIFAR10 ('./formatted_dataset/cifar', transform=transform_image,  train=True, download=False)
    reals = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train(reals, model_discriminator, model_generator, discriminator_loss, generator_loss ,  discriminator_optimizer, generator_optimizer)
    test(model_generator)
#8. Run the main function!
main()


