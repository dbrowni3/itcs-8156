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



        #the forget gate weights and bias 
        #(for f_t = sig(wf1 x_t + wf2 h_{t-1} + b_f))
        self.wf1 ,self.wf2, self.bf = self.initWeights(shape_w1, shape_w2, mean, std)

        #the input gate weights and bias
        #(for i_t = sig(wi1 x_t + wi2 h_{t-1} + b_i))
        self.wi1, self.wi2, self.bi = self.initWeights(shape_w1, shape_w2, mean, std) 
        #the output gate weights and bias
        #(for o_t = sig(wo1 x_t + wo2 h_{t-1} + b_o))
        self.wo1, self.wo2, self.bo = self.initWeights(shape_w1, shape_w2, mean, std)
        #the candidate context weights and bias
        #(for c^'_t = sig(wcc1 x_t + wcc2 h_{t-1} + bc_c))
        self.wcc1, self.wcc2, self.bcc = self.initWeights(shape_w1, shape_w2, mean, std)

        self.whq = nn.Parameter(torch.normal(mean=mean,
                                             std=std,
                                             size=(num_hiddens, num_out)),
                                requires_grad=True,
                                )
        self.bq = nn.Parameter(torch.tensor(0.0),
                        requires_grad=True,
                        )

        # print('wf1 \n', self.wf1.shape, '\n wf2 \n', self.wf2.shape, '\n bf \n', self.bf.shape)
        # print('wi1 \n', self.wi1.shape, '\n wi2 \n', self.wi2.shape, '\n bi \n', self.bi.shape)
        # print('wcc1 \n', self.wf1.shape, '\n wcc2 \n', self.wf2.shape, '\n bf \n', self.bcc.shape)
        # print('wo1 \n', self.wo1.shape, '\n wo2 \n', self.wo2.shape, '\n bo \n', self.bo.shape)

    def initWeights(self, shape_w1, shape_w2, mean, std):
        w1 = nn.Parameter(torch.normal(mean=mean,std=std, size=shape_w1),
                                requires_grad=True,
                                )
        w2 = nn.Parameter(torch.normal(mean=mean,std=std,size=shape_w2),
                                requires_grad=True,
                                )
        bias = nn.Parameter(torch.zeros(self.num_hiddens),
                                requires_grad=True,
                                )
        return w1, w2, bias
    
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
            n = torch.asinh()
            y1 = (B*X) + n(B*X)



        else:
            raise Exception('No activation function specified')
        
        return y1




    def unit(self, val_in, long_mem, short_mem):
        '''
        INPUTS:
            val_in - input into this step of the unit x_t

            long_mem - the long term memory at this step

            short_mem - the short term memory at this step
        OUTPUTS:

        
        '''
        val_in = val_in.float()

        i_t = self.actfunc((self.wi1@val_in)+(self.wi2@short_mem)+(self.bi), self.act_name_1)

        f_t = self.actfunc((self.wf1@val_in)+(self.wf2@short_mem)+(self.bf), self.act_name_1)
        
        o_t = self.actfunc((self.wo1@val_in)+(self.wo2@short_mem)+(self.bo), self.act_name_1)

        cc_t = self.actfunc((self.wcc1@val_in)+(short_mem@short_mem)+(self.bcc), self.act_name_2)

        

        # print('f_t ', f_t)
        # print('i_t ', i_t)
        # print('cc_t ', cc_t)
        # print('o_t ', o_t)

        #update the long term memory (c_t)
        c_t = (f_t*long_mem) + (i_t*cc_t)

        #update the short term memory (h_t)
        h_t = o_t*self.actfunc(c_t, self.act_name_2)
        # print('update_short_mem.shape, ', update_short_mem.shape)

        #for multiple inputs we not do a final layer that is just linear
        # Y = (h_t@self.whq) + self.bq
        Y = h_t

        return [Y, c_t, h_t]

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
            # long_mem[ii+1], short_mem[ii+1] = self.unit(input[ii], 
            #                                         long_mem[ii], 
            #                                         short_mem[ii],
            #                                         )
            # print(input[:,ii])
            Y, long_mem, short_mem = self.unit(input[:,ii], 
                                                    long_mem, 
                                                    short_mem,
                                                    )
        return Y


    def configure_optimizers(self):
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