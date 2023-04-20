import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger as TensorBoardLogger
import tensorboard

import numpy as np
import shap



class BasicLSTM(pl.LightningModule):

    def __init__(self, num_feat, num_hiddens, num_out, lr):
        '''
        num_feat - number of features input into the model
        '''
        super().__init__()
        self.num_feat = num_feat
        self.num_hiddens = num_hiddens
        self.num_out = num_out

        self.lr = lr

        shape_w1 = (num_hiddens,num_feat)
        shape_w2 = (num_hiddens,num_hiddens)
        
        mean = 0.0
        std = 1.0

        #the forget gate weights and bias 
        #(for f_t = sig(wf1 x_t + wf2 h_{t-1} + b_f))
        self.wf1 ,self.wf2, self.bf, self.bf2 = self.initWeights(shape_w1, shape_w2, mean, std)

        #the input gate weights and bias
        #(for i_t = sig(wi1 x_t + wi2 h_{t-1} + b_i))
        self.wi1, self.wi2, self.bi, self.bi2 = self.initWeights(shape_w1, shape_w2, mean, std) 
        #the output gate weights and bias
        #(for o_t = sig(wo1 x_t + wo2 h_{t-1} + b_o))
        self.wo1, self.wo2, self.bo, self.bo2 = self.initWeights(shape_w1, shape_w2, mean, std)
        #the candidate context weights and bias
        #(for c^'_t = sig(wcc1 x_t + wcc2 h_{t-1} + bc_c))
        self.wcc1, self.wcc2, self.bcc, self.bcc2 = self.initWeights(shape_w1, shape_w2, mean, std)

        # self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0 = self.initweights_matrics(shape_w1, shape_w2, mean, std)

        # print('wf1 \n', self.wf1.shape, '\n wf2 \n', self.wf2.shape, '\n bf \n', self.bf.shape)
        # print('wi1 \n', self.wi1.shape, '\n wi2 \n', self.wi2.shape, '\n bi \n', self.bi.shape)
        # print('wcc1 \n', self.wf1.shape, '\n wcc2 \n', self.wf2.shape, '\n bf \n', self.bcc.shape)
        # print('wo1 \n', self.wo1.shape, '\n wo2 \n', self.wo2.shape, '\n bo \n', self.bo.shape)

    def initweights_matrics(self, shape_w1, shape_w2, mean, std):
        w1 = nn.Parameter(torch.normal(mean=mean,std=std, size=(4,shape_w1)),
                                requires_grad=True,
                                )
        
        w2 = nn.Parameter(torch.normal(mean=mean,std=std,size=(4,shape_w2)),
                        requires_grad=True,
                        )
        b1 = nn.Parameter(torch.zeros(self.num_hiddens*4),
                                requires_grad=True,
                                )
        
        b2 = nn.Parameter(torch.zeros(self.num_hiddens*4),
                                requires_grad=True,
                                )
        
        return w1, w2, b1, b2

    def initWeights(self, shape_w1, shape_w2, mean, std):
        w1 = nn.Parameter(torch.normal(mean=mean,std=std, size=shape_w1),
                                requires_grad=True,
                                )
        w2 = nn.Parameter(torch.normal(mean=mean,std=std,size=shape_w2),
                                requires_grad=True,
                                )
        b1 = nn.Parameter(torch.zeros(self.num_hiddens),
                                requires_grad=True,
                                )
        
        b2 = nn.Parameter(torch.zeros(self.num_hiddens),
                                requires_grad=True,
                                )
        return w1, w2, b1, b2


    def unit(self, val_in, long_mem, short_mem):
        '''
        INPUTS:
            val_in - input into this step of the unit x_t

            long_mem - the long term memory at this step

            short_mem - the short term memory at this step
        OUTPUTS:

        
        '''
        if (isinstance(val_in, np.ndarray) == False):
            val_in = val_in.float()

        i_t = torch.sigmoid((self.wi1@val_in)+(self.bi)+(self.wi2@short_mem)+(self.bi2))

        f_t = torch.sigmoid((self.wf1@val_in)+(self.bf)+(self.wf2@short_mem)+(self.bf2))
        
        o_t = torch.sigmoid((self.wo1@val_in)+(self.bo)+(self.wo2@short_mem)+(self.bo2))

        cc_t = torch.tanh((self.wcc1@val_in)+(self.bcc)+(self.wcc2@short_mem)+(self.bcc2))

        #update the long term memory (c_t)
        c_t = (f_t*long_mem) + (i_t*cc_t)

        #update the short term memory (h_t)
        h_t = o_t*torch.tanh(c_t)
        
        return [c_t, h_t]
    
    # def unit(self, val_in, long_mem, short_mem):
    #     '''
    #     INPUTS:
    #         val_in - input into this step of the unit x_t

    #         long_mem - the long term memory at this step

    #         short_mem - the short term memory at this step
    #     OUTPUTS:

        
    #     '''
    #     if (isinstance(val_in, np.ndarray) == False):
    #         val_in = val_in.float()



    #     i_t = torch.sigmoid((self.weight_ih_l0[0,:]@val_in)+(self.bi)+(self.weight_hh_l0[0]@short_mem)+(self.bi2))

    #     f_t = torch.sigmoid((self.weight_ih_l0[1,:]@val_in)+(self.bf)+(self.weight_hh_l0[1]@short_mem)+(self.bf2))
        
    #     cc_t = torch.tanh((self.weight_ih_l0[2,:]@val_in)+(self.bcc)+(self.weight_hh_l0[2]@short_mem)+(self.bcc2))

    #     o_t = torch.sigmoid((self.weight_ih_l0[3,:]@val_in)+(self.bo)+(self.weight_hh_l0[3]@short_mem)+(self.bo2))

    #     #update the long term memory (c_t)
    #     c_t = (f_t*long_mem) + (i_t*cc_t)

    #     #update the short term memory (h_t)
    #     h_t = o_t*torch.tanh(c_t)
        
    #     return [c_t, h_t]


    def forward(self, input):
        '''
        in order case input should be an array with multiple inputs for the model.
        The columns are the features and the rows are the days
        '''

        n_seq = np.shape(input)[-1]

        long_mem = torch.zeros(self.num_hiddens)
        short_mem = torch.zeros(self.num_hiddens)
        
        h_hist=np.zeros(n_seq+1)
        c_hist=np.zeros(n_seq+1)
        x_hist=np.zeros((input.shape[0], n_seq))
        h_hist[0] = short_mem
        c_hist[0] = long_mem

        

        for ii in range(0,n_seq):

            long_mem, short_mem = self.unit(input[:,ii], 
                                                    long_mem, 
                                                    short_mem,
                                                    )

            h_hist[ii+1] = short_mem.detach().numpy()

            c_hist[ii+1] = long_mem.detach().numpy()
            x_hist[:,ii] = input[:,ii].numpy()

        return short_mem, h_hist, c_hist, x_hist


    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_indx):

        input_i, label_i = batch
        output_i = self.forward(input_i[0])[0]

        loss = (output_i - label_i)**2

        self.log("training loss", loss, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])[0]

        test_loss = (output_i - label_i)**2

        self.log("test_loss", test_loss, on_step=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        return self(batch)
    
    
    def makepred(self, dataset):
        y = []
        t = []

        h_hist=[]
        c_hist=[]
        x_hist=[]

        for ii in dataset:
            feat, lab = ii

            y_i, h_i, c_i, x_i = self.forward(feat)

            y_i = (y_i.detach().numpy())

            y.append(y_i)
            t.append(lab.numpy())
            h_hist.append(h_i)
            c_hist.append(c_i)
            x_hist.append(x_i)


        return y, t, h_hist, c_hist, x_hist

        
    def makepredshap(self, x_input):
        targets = np.zeros((np.shape(x_input)[0],1))

        features = torch.from_numpy(x_input)    
        targets = torch.from_numpy(targets) 

        dataset = TensorDataset(features, targets)
        print('print shap dataset ', dataset)

        for ii in dataset:
            feat, lab = ii

            y_i, h_i, c_i, x_i = self.forward(feat)

            y = (y_i.detach().numpy())

        return y

class matrixLSTM(pl.LightningModule):

    def __init__(self, num_feat, num_hiddens, num_out, lr):
        '''
        num_feat - number of features input into the model
        '''
        super().__init__()
        self.num_feat = num_feat
        self.num_hiddens = num_hiddens
        self.num_out = num_out

        self.lr = lr

        shape_w1 = (num_hiddens,num_feat)
        shape_w2 = (num_hiddens,num_hiddens)
        
        mean = 0.0
        std = 1.0


        # self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0 = self.initweights_matrics(shape_w1, shape_w2, mean, std)
        self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0 = self.initweights_matrics(shape_w1, shape_w2, mean, std)

        # print('wf1 \n', self.wf1.shape, '\n wf2 \n', self.wf2.shape, '\n bf \n', self.bf.shape)
        # print('wi1 \n', self.wi1.shape, '\n wi2 \n', self.wi2.shape, '\n bi \n', self.bi.shape)
        # print('wcc1 \n', self.wf1.shape, '\n wcc2 \n', self.wf2.shape, '\n bf \n', self.bcc.shape)
        # print('wo1 \n', self.wo1.shape, '\n wo2 \n', self.wo2.shape, '\n bo \n', self.bo.shape)

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
        
        # b2 = nn.Parameter(torch.zeros(self.num_hiddens*4),
        #                         requires_grad=True,
        #                         )
        
        return w1, w2, b1

  
    def unit(self, val_in, long_mem, short_mem):
        '''
        INPUTS:
            val_in - input into this step of the unit x_t

            long_mem - the long term memory at this step

            short_mem - the short term memory at this step
        OUTPUTS:

        
        '''
        if (isinstance(val_in, np.ndarray) == False):
            val_in = val_in.float()



        i_t = torch.sigmoid((self.weight_ih_l0[0,:]@val_in)+(self.bias_ih_l0[0])+(self.weight_hh_l0[0]@short_mem))

        f_t = torch.sigmoid((self.weight_ih_l0[1,:]@val_in)+(self.bias_ih_l0[1])+(self.weight_hh_l0[1]@short_mem))
        
        cc_t = torch.tanh((self.weight_ih_l0[2,:]@val_in)+(self.bias_ih_l0[2])+(self.weight_hh_l0[2]@short_mem))

        o_t = torch.sigmoid((self.weight_ih_l0[3,:]@val_in)+(self.bias_ih_l0[3])+(self.weight_hh_l0[3]@short_mem))
        # i_t = torch.sigmoid((self.weight_ih_l0[0,:]@val_in)+(self.bias_ih_l0[0])+(self.weight_hh_l0[0]@short_mem)+(self.bias_hh_l0[0]))

        # f_t = torch.sigmoid((self.weight_ih_l0[1,:]@val_in)+(self.bias_ih_l0[1])+(self.weight_hh_l0[1]@short_mem)+(self.bias_hh_l0[1]))
        
        # cc_t = torch.tanh((self.weight_ih_l0[2,:]@val_in)+(self.bias_ih_l0[2])+(self.weight_hh_l0[2]@short_mem)+(self.bias_hh_l0[2]))

        # o_t = torch.sigmoid((self.weight_ih_l0[3,:]@val_in)+(self.bias_ih_l0[3])+(self.weight_hh_l0[3]@short_mem)+(self.bias_hh_l0[3]))

        #update the long term memory (c_t)
        c_t = (f_t*long_mem) + (i_t*cc_t)

        #update the short term memory (h_t)
        h_t = o_t*torch.tanh(c_t)
        
        return [c_t, h_t]


    def forward(self, input):
        '''
        in order case input should be an array with multiple inputs for the model.
        The columns are the features and the rows are the days
        '''

        n_seq = np.shape(input)[-1]

        long_mem = torch.zeros(self.num_hiddens)
        short_mem = torch.zeros(self.num_hiddens)
        

        for ii in range(0,n_seq):

            long_mem, short_mem = self.unit(input[:,ii], 
                                                    long_mem, 
                                                    short_mem,
                                                    )

            return short_mem


    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_indx):

        input_i, label_i = batch
        output_i = self.forward(input_i[0])[0]

        loss = (output_i - label_i)**2

        self.log("training loss", loss, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])[0]

        test_loss = (output_i - label_i)**2

        self.log("test_loss", test_loss, on_step=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        return self(batch)
    
    
    def makepred(self, dataset):
        y = []
        t = []

        for ii in dataset:
            feat, lab = ii

            y_i = self.forward(feat)

            y_i = (y_i.detach().numpy())

            y.append(y_i)
            t.append(lab.numpy())

        return y, t

   

def test_training():
    ## create the training data for the neural network.
    inputs = torch.tensor([[[0., 0.5, 0.25, 1.]], [[1., 0.5, 0.25, 1.]]])
    labels = torch.tensor([0., 1.])

    dataset = TensorDataset(inputs, labels) 
    dataloader = DataLoader(dataset)

    mdl_simple = BasicLSTM(num_feat=1, num_hiddens=1, num_out=1, lr=0.01)
    print("Company A: Observed = 0, Predicted =", 
        mdl_simple(torch.tensor([[0., 0.5, 0.25, 1.]]))[0].detach())
    print("Company B: Observed = 1, Predicted =", 
        mdl_simple(torch.tensor([[1., 0.5, 0.25, 1.]]))[0].detach())


    logger = TensorBoardLogger("lightning_logs", name="simpleModel")

    trainer_simple = pl.Trainer(max_epochs=1000,logger=logger) # with default learning rate, 0.001 (this tiny learning rate makes learning slow)
    trainer_simple.fit(mdl_simple, train_dataloaders=dataloader)

    print("Company A: Observed = 0, Predicted =", 
      mdl_simple(torch.tensor([[0., 0.5, 0.25, 1.]]))[0].detach())
    print("Company B: Observed = 1, Predicted =", 
        mdl_simple(torch.tensor([[1., 0.5, 0.25, 1.]]))[0].detach())
    
    print('saving model')

    torch.save(mdl_simple.state_dict(), './lstm/simple_model_testing')



def test_outputs():
    output_model = BasicLSTM(num_feat=1, num_hiddens=1, num_out=1, lr=0.01)

    output_model.load_state_dict(torch.load('./lstm/simple_model_testing'))

    print('Loaded Model')

    # print("Company A: Observed = 0, Predicted =", 
    #     output_model(torch.tensor([[0., 0.5, 0.25, 1.]]))[0].detach())
    # print("Company B: Observed = 1, Predicted =", 
    #     output_model(torch.tensor([[1., 0.5, 0.25, 1.]]))[0].detach())
    
    ## create the training data for the neural network.
    inputs = torch.tensor([[[0., 0.5, 0.25, 1.]], [[1., 0.5, 0.25, 1.]]])
    labels = torch.tensor([0., 1.])

    dataset = TensorDataset(inputs, labels) 
    dataloader = DataLoader(dataset)

    y, t, h_hist, c_hist, x_hist = output_model.makepred(dataset)

    print('short mem ', y)
    print('label ', t)
    print('h history ', h_hist)
    print('c history ', c_hist)
    print('x history ', x_hist)




def getstockdata():
    import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
    sys.path.insert(1, r'D:\Documents\GitHub\itcs-8156\utils')
    # sys.path.insert(1, r'H:\My Drive\SP23\ML\envs\groupWork\utils')


    from preprocessing import (market_prepro,
                            lstm_timeseries_feat_and_targ,
                            )
    # st = "Stocks"
    st = "ETFs"

    #Input stock name
    sn = "aadr" 
    f = r'D:\Desktop\College Spring 2023\machineLearning\project\coding\data'
    # f = r'H:\My Drive\stockMarket_data'
    X_train, X_test, T_train, T_test = market_prepro(f,st,sn,False,splitdata=True, stdzr='minmax')

    # X,T = market_prepro(f,st,sn,False,splitdata=False)

    #number of days as features
    day_feat = 2

    #number of days to use as features
    day_targ = 1
    day_targ = day_targ - 1

    # dl_train, ds_train = lstm_timeseries_feat_and_targ(X_train[['Open','Low']], T_train, 4, 1,None)
    # dl_test, ds_test = lstm_timeseries_feat_and_targ(X_test[['Open','Low']], T_test, 4, 1,  None)

    dl_train, ds_train = lstm_timeseries_feat_and_targ(X_train, T_train, day_feat, day_targ, [ 'Year', 'Month' ,'Day_date', 'Day'])
    dl_test, ds_test = lstm_timeseries_feat_and_targ(X_test, T_test, day_feat, day_targ, [ 'Year', 'Month' ,'Day_date', 'Day'])
    print('ds_train')
    print(ds_train[0])



    return dl_train, ds_train, dl_test, ds_test

def trainstockmodel():

    dl_train, ds_train, dl_test, ds_test = getstockdata()

    # mdl_stock = BasicLSTM(num_feat=7, num_hiddens=1, num_out=1, lr=0.01)

    # logger = TensorBoardLogger("lightning_logs", name="market")

    # trainer = pl.Trainer(max_epochs=10,logger=logger) # with default learning rate, 0.001 (this tiny learning rate makes learning slow)
    # trainer.fit(mdl_stock, train_dataloaders=dl_train)

    # torch.save(mdl_stock.state_dict(), './lstm/stock_model')



def testshap():

    
    mdl_stock = BasicLSTM(num_feat=7, num_hiddens=1, num_out=1, lr=0.01)

    mdl_stock.load_state_dict(torch.load('./lstm/stock_model'))

    print('Loaded Model')

    dl_train, ds_train, dl_test, ds_test = getstockdata()

    f,t = ds_test[0:2]

    f = f.numpy()
    t = t.numpy()

    # data_test = next(iter(dl_test))[0].numpy()
    print('data test \n', f)
    print('targ \n', t)





    explainer = shap.KernelExplainer(model=mdl_stock.makepredshap, data=f, link="identity")

        
if __name__ == "__main__":
    # test_training()

    # test_outputs()

    trainstockmodel()

    # testshap()
