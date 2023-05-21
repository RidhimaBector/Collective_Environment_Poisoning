import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import argparse
import os
import copy
import pytorch_lightning as pl

from collections import defaultdict
from itertools import count

import os
from os.path import dirname, abspath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Define the Policy_Embedding Network"""
class Policy_Representation(nn.Module):
    def __init__(self, input_size, embedding_size, fc1_units=36, fc2_units=36):
        super(Policy_Representation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, embedding_size)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
#         print(f"x_1 = {x}")
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
    

""" Define the Policy_Imitation Network"""
class Policy_Imitation(nn.Module):
    def __init__(self, input_size, action_size, fc1_units=36, fc2_units=36):
        super(Policy_Imitation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
    
""" Combine Policy_Representation with Policy_Imitation """
class Encoder_Decoder(nn.Module):
    def __init__(self, embedding, imitation, lr):
        super(Encoder_Decoder, self).__init__()
        self.encoder = embedding
        self.decoder = imitation
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        
    def forward(self, x, n_target_states, states):
        #x = torch.cat((s, a),1)
        z = self.encoder(x)
        
        z = torch.unsqueeze(z, 1)
        x = z.repeat(1,n_target_states,1) #torch.cat(len(s)*[z])
        x = torch.cat((states, x), 2)
        
        x = self.decoder(x)
        return x

    
class AutoEncoder():
    
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, lr):

        self.lr = lr
        self.Encoder = Policy_Representation(enc_in_size, enc_out_size).to(device)
        self.Decoder = Policy_Imitation(dec_in_size, dec_out_size).to(device)
        self.Model = Encoder_Decoder(self.Encoder, self.Decoder, self.lr).to(device)
    
    """ Training the Encoder-Decoder network """
    def Train(self, epoch, n_epoch, n_batch, batch_size):

        self.Model.train()
        for i_epoch in range(n_epoch):

            total_loss = 0.0 #Assuming always 1 epoch of training at a time
            for i_batch in range(n_batch):
                
                n_target_states = 16 #6
                states = np.tile(np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=0),(batch_size,1))
                states = np.expand_dims(states, axis=2)
                actions = np.random.choice([0,1,2,3,4], size=(batch_size,n_target_states,1), replace=True)
                batch = np.concatenate((states, actions),2)
                
                states = torch.FloatTensor(states).to(device) #torch.FloatTensor([states]).view(-1,1).to(device)
                actions = torch.FloatTensor(actions).to(device) #torch.FloatTensor([actions]).view(-1,1).to(device)
                batch = torch.FloatTensor(batch).to(device)
                target = actions.long().reshape(batch_size*n_target_states,1).squeeze(1) #actions.long().view(1,6).squeeze(0)

                """ train """
                # clear the gradients of all optimized variables
                self.Model.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.Model(batch, n_target_states, states)
                # calculate the loss
                loss = self.Model.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.Model.optimizer.step()
                # update running training loss
                #loss_list.append([epoch, i_batch, loss.item()]) #*state.size(0)
                total_loss += loss.item()
                print("Epoch: " + str(epoch) + ", Batch: " + str(i_batch) + ", AE Loss = " + str(loss.item()))

        return total_loss/n_batch
#             print('Epoch: {} \tTraining Loss: {:.6f}'.format(i_epoch+1, train_loss))


    """ Embedding using Encoder network """  
    def Policy_Embedding(self, victim_transitions):
        self.Model.eval() # prep model for *evaluation*
        batch = torch.FloatTensor(victim_transitions).to(device)
        z = self.Encoder(batch).cpu().data.numpy()
        return z
    
    def save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename, map_location=device))  #torch.device('cpu')))
        return load_model
    

class VAE_Model(pl.LightningModule):
    def __init__(self, embedding, imitation, latent_size, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = embedding
        self.decoder = imitation
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) #torch.optim.Adam(self.parameters(), lr=1e-4)
        
        # distribution parameters
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_var = nn.Linear(latent_size, latent_size)

    def configure_optimizers(self):
        torch.optim.SGD(self.parameters(), lr=self.lr) #return torch.optim.Adam(self.parameters(), lr=1e-4) #torch.optim.SGD(model.parameters(), lr=0.002)

    def kl_divergence(self, z, mu, std, n_target_states):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        mu = torch.repeat_interleave(mu,n_target_states,0)
        std = torch.repeat_interleave(std,n_target_states,0)
        z = z.reshape(-1,5)
        
        # 1. define the first two probabilities (in this case Normal for both)
        p_dis = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q_dis = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q_dis.log_prob(z)
        log_pz = p_dis.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, states, actions, n_target_states, batch_size):
        
        batch = torch.cat((states, actions), 2) #torch.FloatTensor(np.concatenate((states, actions),2))

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(batch)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q_distribution = torch.distributions.Normal(mu, std)
        z_samples = q_distribution.sample() #q.rsample() #TODO Sample many z based on same q
        z_samples_repeated = torch.unsqueeze(z_samples, 1).repeat(1,n_target_states,1)
        
        #torch.reshape(z_samples, (batch_size, n_target_states, 5))#z_samples = torch.squeeze(z_samples)
        z = torch.cat((states, z_samples_repeated), 2)

        # decoded
        x_hat = self.decoder(z)
        
        # reconstruction loss
        target = actions.long().reshape(batch_size*n_target_states,1).squeeze(1)
        recon_loss = self.criterion(x_hat, target)
        # kl loss
        kl = self.kl_divergence(z_samples_repeated, mu, std, n_target_states)
        elbo = (kl + recon_loss).mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
        })

        return elbo
    
    
class VAE():
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, lr):
        
        self.lr = lr
        self.latent_size = enc_out_size
        self.Encoder = Policy_Representation(enc_in_size, enc_out_size).to(device)
        self.Decoder = Policy_Imitation(dec_in_size, dec_out_size).to(device)
        self.Model = VAE_Model(self.Encoder, self.Decoder, self.latent_size, self.lr).to(device)

    def Train(self, epoch, n_epoch, n_batch, batch_size):
        
        self.Model.train()
        for i_epoch in range(n_epoch):

            total_loss = 0.0 #Assuming always 1 epoch of training at a time
            for i_batch in range(n_batch):
                
                n_target_states = 16 #6
                states = np.tile(np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=0),(batch_size,1))
                states = np.expand_dims(states, axis=2)
                actions = np.random.choice([0,1,2,3,4], size=(batch_size,n_target_states,1), replace=True)
                #batch = np.concatenate((states, actions),2)
                
                states = torch.FloatTensor(states).to(device) #torch.FloatTensor([states]).view(-1,1).to(device)
                actions = torch.FloatTensor(actions).to(device) #torch.FloatTensor([actions]).view(-1,1).to(device)
                #batch = torch.FloatTensor(batch).to(device)
                #target = actions.long().reshape(batch_size*n_target_states,1).squeeze(1)
                
                """ train """
                # clear the gradients of all optimized variables
                self.Model.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                loss = self.Model(states, actions, n_target_states, batch_size)
                # calculate the loss
                #loss = self.Model.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.Model.optimizer.step()
                # update running training loss
                #loss_list.append([epoch, i_batch, loss.item()]) #*state.size(0)
                total_loss += loss.item()
                print("Epoch: " + str(epoch) + ", Batch: " + str(i_batch) + ", VAE Loss = " + str(loss.item()))
                
        return total_loss/n_batch
    
    def Embedding_Samples(self, victim_transitions, no_of_samples):
        
        self.Model.eval()
        
        # encode x to get the mu and variance parameters
        x = torch.FloatTensor(victim_transitions).to(device)
        x_encoded = self.Model.encoder(x)
        mu, log_var = self.Model.fc_mu(x_encoded), self.Model.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q_dist = torch.distributions.Normal(mu, std)
        z_samples = q_dist.sample((no_of_samples,)) #q.rsample() #TODO Sample many z based on same q
        z_samples = torch.squeeze(z_samples, dim=1)

        return z_samples.cpu().data.numpy()
    
    def save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_VAE")

    def load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename, map_location=device))  #torch.device('cpu')))
        return load_model
    
    
if __name__ == "__main__":
    
    """ Intialization """
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    enc_in_size = 32 # 32 as 16 total states #SEQ_LEN*2 = 5*2
    enc_out_size = 5 # EMBEDDING_SIZE = 5
    dec_in_size = 6 #1+EMBEDDING_SIZE
    dec_out_size = 5 #action_dim 0,1,2,3,4 - 4 for target states that victim has not visited yet
    
    net_args = {
        "enc_in_size": enc_in_size, 
        "enc_out_size": enc_out_size, 
        "dec_in_size": dec_in_size, 
        "dec_out_size": dec_out_size, 
        "lr": 0.00001, #0.001,
    }
    
    vae = VAE(**net_args) #ae = AutoEncoder(**net_args)
    print("VAE Initialized")
    
    epoch = 1
    n_epoch = 1
    n_batch = 500
    batch_size = 512
    vae_loss_list = []
    while epoch > 0:
        vae_loss_list.append([epoch, vae.Train(epoch, n_epoch, n_batch, batch_size)])
        vae.save(f"./{epoch}")
        if(epoch % 20 == 0):
            np.savetxt("VAE_loss_buffer.csv", np.array(vae_loss_list), delimiter=",")
        
        epoch = epoch + 1
    
    """states = np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=1)
    actions = np.random.choice([0,1,2,3,4], size=(16,1), replace=True)
    victim_transitions = np.concatenate((states, actions),1)
    vae.Embedding_Samples(victim_transitions, 1)"""
