import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger as TensorBoardLogger
import tensorboard
import numpy as np

class BasicLSTM(pl.LightningModule):
    
    def __init__(self, num_feat, num_hiddens, num_out, lr, actfn1 = 'Sigmoid', actfn2 = 'tanh'):
        '''
        num_feat - number of features input into the model
        '''
        super().__init__()
        self.num_feat = num_feat
        self.num_hiddens = num_hiddens
        self.num_out = num_out

        self.lr = lr
        self.act_name_1 = actfn1
        self.act_name_2 = actfn2

        shape_w1 = (num_hiddens,num_feat)
        shape_w2 = (num_hiddens,num_hiddens)
        
        mean = 0.0
        std = 1.0

        self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0 = self.initweights_matrics(shape_w1, shape_w2, mean, std)

    def initweights_matrics(self, shape_w1, shape_w2, mean, std):

        w1 = nn.Parameter(torch.normal(mean=mean,std=std, size=(4,shape_w1[1])),
                                requires_grad=True,
                                )
        
        w2 = nn.Parameter(torch.normal(mean=mean,std=std,size=(4,shape_w2[1])),
                        requires_grad=True,
                        )
        b1 = nn.Parameter(torch.zeros(self.num_hiddens*4),
                                requires_grad=True,
                                )
        
        b2 = nn.Parameter(torch.zeros(self.num_hiddens*4),
                                requires_grad=True,
                                )
        
        return w1, w2, b1, b2
    
    
    def actfunc(self, X, act_name, B=0.5):
        '''
        INPUTS:
            X - input into the activation function (originally/defaulting to sigmoid)

            act_name - the name of the desired activation function [str]

        OUTPUTS:
            y1 - the value of the input after being passed through the activation function.
        
        '''


        if act_name == 'ReLU':
            n = nn.ReLU(X)
            y1 = n(X)

        elif act_name == 'LeakyReLU':
            n=nn.LeakyReLU(0.2)
            y1=n(X)

        elif act_name == 'ELU':
            n=nn.ELU()
            y1=n(X)
        
        elif act_name == 'SELU':
            n=nn.SELU()
            y1=n(X)

        elif act_name == 'arcsinh':
            y1 = torch.arcsinh(B*X)

        elif act_name == 'Swish':
            n=nn.SiLU(X)
            y1 = n(X)
        
        elif act_name == 'Sigmoid':
            n = nn.Sigmoid()
            y1 = n(X)      

        elif act_name == 'tanh':
            m = nn.Tanh()
            y1=m(X)  

        elif act_name == 'Softplus':
            m = nn.Softplus()
            y1 = m(X)

        elif act_name == 'Mish':
            m = nn.Softplus()
            n = nn.Tanh()
            y1 =  X* n(m(X))

        elif act_name == 'Comb-H-Sine':
            y1 = (B*X) + torch.arcsinh(B*X)



        #else:
            #raise Exception('No activation function specified')
        
        return y1




    def unit(self, val_in, long_mem, short_mem):
        '''
        INPUTS:
            val_in - input into this step of the unit x_t

            long_mem - the long term memory at this step

            short_mem - the short term memory at this step
        OUTPUTS:
            c_t - long term memory

            h_t - short term memory   
        '''

        if (isinstance(val_in, np.ndarray) == False):
            val_in = val_in.float()

        i_t = self.actfunc((self.weight_ih_l0[0,:]@val_in)+(self.bias_ih_l0[0])+(self.weight_hh_l0[0]@short_mem)+(self.bias_hh_l0[0]), self.act_name_1)

        f_t = self.actfunc((self.weight_ih_l0[1,:]@val_in)+(self.bias_ih_l0[1])+(self.weight_hh_l0[1]@short_mem)+(self.bias_hh_l0[1]), self.act_name_1)
        
        cc_t = self.actfunc((self.weight_ih_l0[2,:]@val_in)+(self.bias_ih_l0[2])+(self.weight_hh_l0[2]@short_mem)+(self.bias_hh_l0[2]), self.act_name_1)

        o_t = self.actfunc((self.weight_ih_l0[3,:]@val_in)+(self.bias_ih_l0[3])+(self.weight_hh_l0[3]@short_mem)+(self.bias_hh_l0[3]), self.act_name_2)


        # print('f_t ', f_t)
        # print('i_t ', i_t)
        # print('cc_t ', cc_t)
        # print('o_t ', o_t)

        #update the long term memory (c_t)
        c_t = (f_t*long_mem) + (i_t*cc_t)

        #update the short term memory (h_t)
        h_t = o_t*self.actfunc(c_t, self.act_name_2)
        # print('update_short_mem.shape, ', update_short_mem.shape)


        return [c_t, h_t]

    def forward(self, input):
        '''
        in order case input should be an array with multiple inputs for the model.
        The columns are the features and the rows are the days
        '''
        n_seq = np.shape(input)[-1]

        # long_mem = torch.zeros(n_seq, requires_grad=False)
        # short_mem = torch.zeros(n_seq, requires_grad=False)
        long_mem = torch.zeros(self.num_hiddens)
        short_mem = torch.zeros(self.num_hiddens)
        
        for ii in range(0,n_seq):
            long_mem, short_mem = self.unit(input[:,ii], 
                                                    long_mem, 
                                                    short_mem,
                                                    )
        return short_mem


    def configure_optimizers(self):
        #return torch.optim.NAdam(self.parameters(), lr=self.lr)
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_indx):

        input_i, label_i = batch
        output_i = self.forward(input_i[0])

        # print(output_i.shape)
        # print(output_i)
        # print(label_i.shape)
        # print(label_i[0,0].shape)
        # print(label_i[0,0])

        loss = (output_i - label_i)**2
        # loss = nn.functional.mse_loss(output_i, label_i)

        # loss = loss.float()

        # self.log("training loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("training loss", loss, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])

        test_loss = (output_i - label_i)**2

        self.log("test_loss", test_loss, on_step=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        return self(batch)