import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image



class SinusoidalPositionEmbeddings(nn.Module):


    def __init__(self, time_emb_dim):

        '''simple initialization that takes in an int (ex. 300) as 
        the time embedding dimension during intialization...'''

        super().__init__()

        #int (ex. 300) - how large the time embedding will be...
        self.dim = time_emb_dim


    def forward(self, time):

        '''not so simple math part...'''

        #time tensor of shape (batch_size); of random integers from 0 to 299...
        t = time
        
        #reshape to (batch_size, 1) to get a sequence of random time positions...
        pos = t.unsqueeze(1)

        #embedding tensor of shape (1, dim); content: integers from 0 to 299...
        i = torch.arange(self.dim).unsqueeze(0)

        #use tensor of i's to generate unique frequencies for every dimension of the positional embedding...
        unique_frequency = 1 / (10000 ** (2 * (i // 2) / self.dim))

        #pos (batch_size, 1) * unique_frequency (1, dim) = angles (batch_size, dim)
        angles = pos * unique_frequency

        #apply sine to even indices and cosine to odd indices...
        pos_embedding = torch.zeros((t.shape[0], self.dim))
        pos_embedding[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_embedding[:, 1::2] = torch.cos(angles[:, 1::2])
        return pos_embedding.to(time.device)
    

    def graph(self, timesteps = 300):

        '''creates a cool visual of the embedding tensors...'''

        #generate embeddings...
        embedding_layer = SinusoidalPositionEmbeddings(self.dim)
        time_steps = torch.arange(timesteps)
        embeddings = embedding_layer(time_steps).cpu().detach().numpy()  

        #plot heatmap...
        plt.figure(figsize = (12, 6))
        sns.heatmap(embeddings, cmap = "coolwarm", xticklabels = 20, yticklabels = 5)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Time Step")
        plt.title("Sinusoidal Positional Embeddings")
        plt.show()



class Block(nn.Module):


    def __init__(self, in_ch, out_ch, time_emb_dim, num_labels, up = False):

        '''bifunctional block of convolutional layers which process image 
        data and time data during the up & down sampling process...'''

        super().__init__()
        
        #create a time embedding multilayer perceptron...
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        #create another multilayer perceptron for image labels...
        self.num_labels = num_labels
        self.label_mlp = nn.Linear(self.num_labels, out_ch)

        #decrease channels by factor of 4 (because of residual connections) and increase H & W by a factor of 2...
        if up:

            #decrease the input + residual channels by a factor of 4...
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding = 1)

            #increase the H & W by a factor of 2...
            self.final = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

        #increase channels by factor of 2 and decrease H & W by a factor of 2...
        else:

            #increase channels by a factor of 2...
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)

            #decrease the H & W by a factor of 2...
            self.final = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        #channels and H & W stay the same; just further processing...
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)

        #batch normalization layers...
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        #activation layer...
        self.relu  = nn.ReLU()


    def forward(self, x, t, label):

        '''block of convolution layers for upsampling or downsampling...'''

        #first convolution...
        h = self.bnorm1(self.relu(self.conv1(x)))

        #time embedding...
        time_emb = self.relu(self.time_mlp(self.time_embedding(t)))

        #extend last 2 dimensions...
        time_emb = time_emb[(..., ) + (None, ) * 2]

        #add time channel...
        h = h + time_emb

        #a single lable is a 1D tensor of int 0 to 2 that gets fed through a mlp like time...
        one_hot = F.one_hot(label, self.num_labels).float()
        l = self.relu(self.label_mlp(one_hot))

        #extend last 2 dimensions and add to rest of data...
        l = l[(..., ) + (None, ) * 2]
        h = h + l

        #second convolution...
        h = self.bnorm2(self.relu(self.conv2(h)))

        #return down or upsample...
        return self.final(h)



class SimpleUnet(nn.Module):


    def __init__(self, num_labels, base_channels = 64, image_channels = 3, time_emb_dim = 300):

        '''cnn with symmetric encoder-decoder structure with skip connections...'''

        super().__init__()

        #R, G, B...
        self.image_channels = image_channels

        #int to initialize the time embedding classes throughout the unet...
        self.time_emb_dim = time_emb_dim

        #start at max image dimension & base channels...
        self.down_channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            )

        #end at max image dimension & base channels...
        self.up_channels = (
            base_channels * 16,
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
            )

        #initial projection...
        self.conv0 = nn.Conv2d(self.image_channels, base_channels, 3, padding = 1)

        #downsample...
        self.downs = nn.ModuleList([Block(
            self.down_channels[i],
            self.down_channels[i + 1],
            self.time_emb_dim,
            num_labels,
            up = False) for i in range(len(self.down_channels) - 1)])

        #upsample...
        self.ups = nn.ModuleList([Block(
            self.up_channels[i],
            self.up_channels[i + 1],
            self.time_emb_dim,
            num_labels,
            up = True) for i in range(len(self.up_channels) - 1)])

        #output layer...
        self.output = nn.Conv2d(base_channels, self.image_channels, 1)


    def forward(self, x, t, label):

        '''take in a batch of images with noise, a time tensor, and a label tensor 
        and output a batch of predicted noise...'''

        #run initial projection...
        x = self.conv0(x)

        #residual inputs are fed to the opposite side of the unet to enable bypassing of down & up sampling...
        residual_inputs = []

        #downsample...
        for down in self.downs:
            x = down(x, t, label)
            residual_inputs.append(x)
            
        #upsample....
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim = 1)
            x = up(x, t, label)

        #return a batch of predicted noise...
        return self.output(x)
    


class DiffusionModel(nn.Module):


    def __init__(self, images, load_checkpoint = '', img_size = 64, start_schedule = 0.0001, 
                 end_schedule = 0.02, timesteps = 300, set_random_seed = True):
        
        '''full diffusion model with integrated unet...'''

        super().__init__()

        #image size...
        self.img_size = img_size

        #minimum bound of betas...
        self.start_schedule = start_schedule

        #maximum bound of betas...
        self.end_schedule = end_schedule

        #maximum number of diffusion itterations...
        self.timesteps = timesteps

        #betas: amount of noise added to the image at timestep; quadratic schedule prevents overdiffusion early on...
        t = torch.linspace(0, 1, timesteps)
        self.betas = start_schedule + (end_schedule - start_schedule) * (t ** 2)

        #alphas: 1 - betas; amount of original image information that is being preserved at every timestep...
        self.alphas = 1 - self.betas

        #alpha_cumprod: a sequence of products where ah[0] = a[0], ah[1] = a[0] * a[1], ah[2] = a[0] * a[1] * a[2] ...
        self.alpha_cumprod = torch.cumprod(self.alphas, axis = 0)

        #function to turn a PIL image into a tensor...
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        #function to turn a tensor back in a PIL image...
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
            transforms.Resize((self.img_size - 4, self.img_size - 4)),
        ])

        #process image dataset...
        self.image_tensors = []
        self.labels = []
        self.l_dict = {}

        #converts images to a list of image tensors and cooresponding label codes, and creates a dictionary to convert label to label code...
        for idx, path_label in enumerate(images):

            #convert each jpg into tensor and add to a image dataset...
            pi = Image.open(os.path.join(path_label[0]))
            ti = self.transform(pi)
            self.image_tensors.append(ti)

            #add label codes to a dictionary for later conversion and add codes to label list...
            if path_label[1] not in self.l_dict:
                self.l_dict[path_label[1]] = idx
            self.labels.append(self.l_dict[path_label[1]])

        #initialize unet...
        self.unet = SimpleUnet(len(self.l_dict))

        #option to initialize unet with saved weights...
        if load_checkpoint:
            self.cpp = load_checkpoint
            self.load_state_dict(torch.load(os.path.join(self.cpp, "unet_weights.pth")))

        #for reproducibility...
        if set_random_seed:
            torch.manual_seed(1)
    

    @staticmethod
    def get_index_from_list(values, t, x_shape):

        '''makes indexing easier...'''

        batch_size = t.shape[0]
        result = values.gather(-1, t.cpu())
        return result.reshape(batch_size, * ((1,) * (len(x_shape) - 1))).to(t.device)


    def forward(self, x0, t, device):

        '''take in a batch of images partnered with timesteps 
        and adds noise to each image based on the timestep...'''

        #random number tensor with same shape as x0 (input batch): [batch, channel, height, width]...
        self.noise = torch.randn_like(x0, device = device)

        #alpha_cumprod_t: alpha_cumprod values indexed by t but in 4 dimensional tensor...
        self.alpha_cumprod_tf = self.get_index_from_list(self.alpha_cumprod, t, x0.shape).to(device)

        #intermidiate computation variables...
        self.sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod_tf)
        self.sqrt_one_minus_alpha_cumprod_tf = torch.sqrt(1.0 - self.alpha_cumprod_tf)

        #mean multiplies pixel values by decreasing alpha values and makes the image 'hazy'...
        self.mean = self.sqrt_alpha_cumprod_t * x0

        #variance adds the noise...
        self.variance = self.sqrt_one_minus_alpha_cumprod_tf * self.noise
        
        #return the noised images...
        return self.mean + self.variance, self.noise


    @torch.no_grad()
    def backward(self, x, t, label):

        '''take in a batch of noisy images partnered with labels & 
        timesteps and denoise each image based on the timestep and its label...'''

        #use t to index betas...
        self.betas_t = self.get_index_from_list(self.betas, t, x.shape)

        #math from scientists...
        self.alphas_t = self.get_index_from_list(self.alphas, t, x.shape)
        self.sqrt_recip_alphas_t = 1 / torch.sqrt(self.alphas_t)
        self.alpha_cumprod_tb = self.get_index_from_list(self.alpha_cumprod, t, x.shape)
        self.sqrt_one_minus_alphas_cumprod_tb = torch.sqrt(1 - self.alpha_cumprod_tb)
        self.pred_noise = self.unet(x, t, label)

        #call model to predict the noise then substract the noise from the image...
        self.model_mean = self.sqrt_recip_alphas_t * (x - self.betas_t * self.pred_noise / self.sqrt_one_minus_alphas_cumprod_tb)

        #posterior variance: the uncertainty in the reverse process when predicting noise...
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[: -1], (1,0), value = 1)
        self.posterior_variance = self.betas_t * (1 - self.get_index_from_list(self.alpha_cumprod_prev, t, x.shape)) / (1 - self.alpha_cumprod_tb)

        #return the denoised images...
        if t.item() == 0:
            return self.model_mean
        else:
            noise = torch.randn_like(x)
            self.model_variance = torch.sqrt(self.posterior_variance) * noise
            return self.model_mean + self.model_variance


    def train_model(self, epochs = 100, batch_size = 6, lr = 1e-5, device = 'cpu', checkpoint_path = '', loss_update = 5):

        '''trains unet on image dataset...'''

        #ensure normalization layers are using batch statistics...
        self.unet.train()

        #contains all images in tensor form with corresponding label tensors...
        dataset = TensorDataset(torch.stack(self.image_tensors), torch.tensor(self.labels))
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True) 

        #create optimizer...
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr = lr)

        #training loop...
        print(f'Updating {(sum(p.numel() for p in self.unet.parameters()))/1e6:.2f} M Parameters...')
        loss_list = []
        for epoch in range(epochs):
            ell = []
            for batch, label in train_loader:

                #create a tensor of random time indices...
                t = torch.randint(0, self.timesteps, (batch.shape[0],)).long().to(device)

                #forward pass to define noisy images and corresponding ground truth noise...
                batch_noisy, noise = self.forward(batch, t, device = device)

                #feed noisy images into unet to get predicted noise...
                predicted_noise = self.unet(batch_noisy, t, label)

                #evaluate loss...
                loss = torch.nn.functional.mse_loss(noise, predicted_noise)

                #update parameters...
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ell.append(loss.item())

            #update loss with mean epoch loss...
            loss_list.append(np.mean(ell))

            #display training progress...
            if epoch % loss_update == 0:
                max_bar = 50
                ratio = epoch / epochs
                print(f"Training... |{'█' * (round(max_bar * ratio)) + '▒' * (max_bar - round(max_bar * ratio))}| Epoch: {epoch:>4.0f}/{epochs} | Loss: {np.mean(ell):.5f} |")
                if checkpoint_path:
                    torch.save(self.state_dict(), os.path.join(checkpoint_path, "unet_weights.pth"))

        #report final loss...
        self.final_loss = np.mean(loss_list)
        print(f"Complete!   |{'█' * max_bar}| Epoch: {epochs:>4.0f}/{epochs} | Loss: {self.final_loss:.5f} |")


    def generate_image(self, label, save_folder, show_graph = False, device = 'cpu'):
        
        '''generate an image from random noise given a label...'''

        #gather params and create a plot for display...
        img_shape = (self.img_size, self.img_size)
        f, ax = plt.subplots(1, 3, figsize = (10, 10))
        shown_timesteps = torch.linspace(self.timesteps - 1, 0, steps = 3, dtype = torch.long).tolist()
        l = torch.tensor([self.l_dict[label]])
        
        #use saved running means and variances from training normalization layers...
        self.unet.eval()
        print('\nGenerating Image...')
        with torch.no_grad():

            #create a image of pure noise...
            img = torch.randn((1, 3) + img_shape).to(device)

            #starting from the last step in timesteps...
            for i in reversed(range(self.timesteps)):

                #throw the timestep & label and the noisy image into the diffusion model to get the denoised image...
                t = torch.full((1,), i, dtype = torch.long, device = device)
                img = self.backward(img, t, l)
                if i % 10 == 0:
                    max_bar = 50
                    ratio = (self.timesteps - i) / self.timesteps
                    print(f"Denoising Image... |{'█' * (round(max_bar * ratio)) + '▒' * (max_bar - round(max_bar * ratio))}| Timestep: {(self.timesteps - i):>4.0f}/{self.timesteps} |")

                #show what the image looks like at the certain timestep...
                if i in shown_timesteps:
                    idx = shown_timesteps.index(i)
                    ax[idx].imshow(self.reverse_transform(img[0]))
                    ax[idx].set_title(f't = {i}', fontsize = 14)
        
        #save the image as a jpg into the save folder...
        gen = self.reverse_transform(img[0])
        path = os.path.join(save_folder, f'{label}.jpg')
        gen.save(path, format = 'JPEG')
        
        #display diffusion process...
        if show_graph:
            plt.tight_layout()
            plt.show()