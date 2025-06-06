import torch
import torch.nn as nn
import numpy as np
from matrix_factorization_noisy_ori import *
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class NCF_BaseModel_3(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel_3, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k, bias = True)
        self.linear_2 = torch.nn.Linear(self.embedding_k, self.embedding_k, bias = True)
        self.linear_3 = torch.nn.Linear(self.embedding_k, 1, bias = True)
        self.relu = torch.nn.ReLU()
        # self.sigmoid = torch.nn.Sigmoid()

        # self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        x=self.linear_1(z_emb)
        x=self.relu(x)
        x=self.linear_2(x)
        x=self.relu(x)
        x=self.linear_3(x)

        out = x
        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class MF_BV_Multi_bin_JL_G(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, batch_size_prop=1024, monte_sample_num=10, rho=None, rho_given=None, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.monte_sample_num = monte_sample_num

        self.normal = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        self.rho_num = rho_given
        if rho is None:
            self.rho_v = Parameter(torch.Tensor([0]), requires_grad=True)
        else:
            self.rho_v = Parameter(torch.Tensor([rho]), requires_grad=True)

        self.sigmoid = torch.nn.Sigmoid()
        # self.xent_func = torch.nn.MSELoss()
        self.tanh = torch.nn.Tanh()
        self.xent_func = torch.nn.BCEWithLogitsLoss()

    def rho(self):
        if self.rho_num is None:
            return self.tanh(self.rho_v)
            # return torch.clip(self.rho_v, min=-1, max=0.99)
        else:
            return torch.Tensor([self.rho_num]).cuda()
        # return torch.Tensor([0]).cuda()
    def pre_fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                # print(selected_idx)
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model(sub_x, False)
                xent_loss = self.xent_func(pred,sub_y)
                # xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)))   
                #loss = xent_loss

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def fit(self, x, y,
            num_epoch=1000, lr=0.05, lamb=0,
            tol=1e-4, verbose=False, alpha=None, G=1):

        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(
            self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_all = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(
            self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # optimizer = torch.optim.Adam(
        #     self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)

        # optimizer_pred = torch.optim.Adam(
        #     self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        # optimizer_prop = torch.optim.Adam(
        #     self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        # g_y_all = sps.csr_matrix((g_y_all, (x[:, 0], x[:, 1])), shape=(
        #     self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # g_o_all = sps.csr_matrix((g_o_all, (x[:, 0], x[:, 1])), shape=(
        #     self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        last_loss = 1e9

        num_sample = len(x)

        epsilon_all = self.normal.sample(
                    (self.num_users*self.num_items, self.monte_sample_num)).squeeze(-1)

        total_batch = num_sample // self.batch_size
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            # epoch_loss_o1=0
            # epoch_loss_o0=0
            # epoch_loss_o1_y=0
            # epoch_loss_o1_phi=0

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)


            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size *
                                       idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                
                # sub_o = torch.Tensor(obs[selected_idx]).cuda()
                # sub_y = torch.Tensor(y[selected_idx]).unsqueeze(-1).cuda()
                
                # idx_o1=(sub_o==1).detach().cpu().numpy()
                # idx_o0=(sub_o==0).detach().cpu().numpy()

                sub_y = torch.Tensor(y[selected_idx]).cuda()
                sub_epsilon = epsilon_all[sub_x[:,0]*self.num_items+sub_x[:,1]]

                x_sampled_idx=ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_sampled_idx]
                sub_o = obs[x_sampled_idx]
                x_sampled_o1=x_sampled[sub_o==1]
                x_sampled_o0=x_sampled[sub_o==0]
                y_sampled = torch.Tensor(y_all[x_sampled_idx][sub_o==1]).cuda()

                # update gy using o=1 data

                g_y= self.prediction_model.forward(sub_x, False).unsqueeze(-1)
                g_o = self.propensity_model.forward(sub_x, False).unsqueeze(-1).detach()


                pos_idx = sub_epsilon + g_o > 0

                logphi=self.normal.cdf((g_y+self.rho()*sub_epsilon)/torch.sqrt(1-self.rho()**2)) * pos_idx.detach()
                e_logphi=torch.mean(logphi,axis=1)
                logphi_o = self.normal.cdf(g_o).squeeze(-1)
                
                y_loss = -torch.mean(torch.log(torch.clip(e_logphi,min=1e-5))*sub_y+torch.log(torch.clip(logphi_o-e_logphi,min=1e-5))*(1-sub_y))

                # calculate MF losss
                mf_loss=self.xent_func(g_y,sub_y.reshape(g_y.shape))
                if alpha is None:
                    y_loss = y_loss*torch.clip(self.rho().detach(),min=0.01,max=0.99)+torch.clip(1-self.rho().detach(),min=0.01,max=0.99)*mf_loss
                else:
                    y_loss = y_loss*alpha+(1-alpha)*mf_loss


                optimizer.zero_grad()
                y_loss.backward()
                optimizer.step()

                # update g_o

                g_y = self.prediction_model.forward(x_sampled_o1, False).unsqueeze(-1).detach()
                g_o = self.propensity_model.forward(x_sampled, False).unsqueeze(-1)

                sub_epsilon = epsilon_all[x_sampled_idx][sub_o==1]

                pos_idx = sub_epsilon + g_y > 0

                logphi=self.normal.cdf((g_o[sub_o==1]+self.rho()*sub_epsilon)/torch.sqrt(1-self.rho()**2)) * pos_idx.detach()
                e_logphi=torch.mean(logphi,axis=1)
                logphi_o = self.normal.cdf(g_o[sub_o==1]).squeeze(-1)
                # logphi_o0=self.normal.cdf(-g_o[idx_o0])
                
                o_loss1 = -torch.sum(torch.log(torch.clip(e_logphi,min=1e-5))*y_sampled+torch.log(torch.clip(logphi_o-e_logphi,min=1e-5))*(1-y_sampled))
                o_loss0=-torch.sum(torch.clip(self.normal.cdf(-g_o[sub_o==0]),min=1e-5).log())
                o_loss=o_loss0+o_loss1
                optimizer.zero_grad()
                o_loss.backward()
                optimizer.step()


                # g_y = torch.Tensor(g_y_all[selected_idx][idx_o1]).cuda()
                # g_o = torch.Tensor(g_o_all[selected_idx]).cuda()
                # log_o1_y=-((sub_y-g_y)**2)/2
                # log_o1_phi=torch.clip(self.normal.cdf((g_o[idx_o1]+self.rho()*(sub_y-g_y))/torch.sqrt(1-self.rho()**2)),min=1e-5).log()
                # log_o1=log_o1_y+log_o1_phi
                # log_o0=torch.clip(self.normal.cdf(-g_o[idx_o0]),min=1e-5).log()
                

                # # xent_loss_o1_y=-torch.sum(log_o1_y)
                # # xent_loss_o1_phi=-torch.sum(log_o1_phi)
                # xent_loss_o1=-torch.sum(log_o1)
                # xent_loss_o2=-torch.sum(log_o0)

                xent_loss = y_loss+o_loss

                

                # loss = xent_loss

                # if num_epoch>1:
                #     optimizer.zero_grad()
                #     loss.backward()
                #     # print(self.rho.grad)
                #     optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()
                # epoch_loss_o1 += xent_loss_o1.detach().cpu().numpy()
                # epoch_loss_o0 += xent_loss_o2.detach().cpu().numpy()
                # epoch_loss_o1_y += xent_loss_o1_y.detach().cpu().numpy()
                # epoch_loss_o1_phi += xent_loss_o1_phi.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    # print("[MF_BV] epoch:{}, xent_o1:{}, xent_o0:{}, xent_o1_y:{}, xent_o1_phi:{}".format(epoch, epoch_loss_o1, epoch_loss_o0,epoch_loss_o1_y,epoch_loss_o1_phi))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 1 == 0 and verbose:
                print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                # print("[MF_BV] epoch:{}, xent_o1:{}, xent_o0:{}".format(epoch, epoch_loss_o1, epoch_loss_o0))

            if epoch == num_epoch - 1:
                print("[MF_BV] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def fit_no_o(self, x, y, batch_size=2048,
            num_epoch=1000, lr=0.001, lamb=0,
            tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        

        last_loss = 1e9

        num_sample = len(x)


        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size *
                                       idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx]).cuda()

                g_y, _, _ = self.prediction_model.forward(sub_x, True)

                # log_o1=-((sub_y-g_y)**2)/2

                xent_loss = self.xent_func(g_y,sub_y)
                # xent_loss=nn.MSELoss()(g_y,sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                # print(self.rho.grad)
                optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 1 == 0 and verbose:
                print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_BV] Reach preset epochs, it seems does not converge.")
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=1e-4, 
        tol=1e-5, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0


        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                prop = self.sigmoid(prop)
                # propensity score

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")

class MF_BV_Multi_bin_JL_G_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, batch_size_prop=1024, monte_sample_num=10, rho=None, rho_given=None, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.monte_sample_num = monte_sample_num

        self.propensity_model_2 = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation_model = NCF_BaseModel_3(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)

        self.normal = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        self.rho_num = rho_given
        if rho is None:
            self.rho_v = Parameter(torch.Tensor([0]), requires_grad=True)
        else:
            self.rho_v = Parameter(torch.Tensor([rho]), requires_grad=True)

        self.sigmoid = torch.nn.Sigmoid()
        # self.xent_func = torch.nn.MSELoss()
        self.tanh = torch.nn.Tanh()
        self.xent_func = torch.nn.BCEWithLogitsLoss()
        self.x_all = generate_total_sample(self.num_users, self.num_items)

    def rho(self):
        if self.rho_num is None:
            return self.tanh(self.rho_v)
            # return torch.clip(self.rho_v, min=-1, max=0.99)
        else:
            return torch.Tensor([self.rho_num]).cuda()
        # return torch.Tensor([0]).cuda()
    def pre_fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                # print(selected_idx)
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model(sub_x, False)
                xent_loss = self.xent_func(pred,sub_y)
                # xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)))   
                #loss = xent_loss

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def fit(self, x, y,
            num_epoch=1000, lr=0.05, lamb=0,
            tol=1e-4, verbose=False, alpha=None, G=1, gamma=0.015):

        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(
            self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_all = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(
            self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # optimizer = torch.optim.Adam(
        #     self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        num_sample = len(x)

        epsilon_all = self.normal.sample(
                    (self.num_users*self.num_items, self.monte_sample_num)).squeeze(-1)

        total_batch = num_sample // self.batch_size
        x_all = self.x_all
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)


            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size *
                                       idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]

                sub_y = torch.Tensor(y[selected_idx]).cuda()
                sub_epsilon = epsilon_all[sub_x[:,0]*self.num_items+sub_x[:,1]]

                x_sampled_idx=ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_sampled_idx]
                sub_o = obs[x_sampled_idx]
                x_sampled_o1=x_sampled[sub_o==1]
                x_sampled_o0=x_sampled[sub_o==0]
                y_sampled = torch.Tensor(y_all[x_sampled_idx][sub_o==1]).cuda()

                # update gy using o=1 data

                g_y= self.prediction_model.forward(sub_x, False).unsqueeze(-1)
                g_o = self.propensity_model.forward(sub_x, False).unsqueeze(-1).detach()

                logphi_o = self.normal.cdf(g_o).squeeze(-1)
                pos_idx = sub_epsilon + g_o > 0

                logphi=self.normal.cdf((g_y+self.rho()*sub_epsilon)/torch.sqrt(1-self.rho()**2)) * pos_idx.detach()
                e_logphi=torch.mean(logphi,axis=1)
                # logphi_o = self.normal.cdf(g_o).squeeze(-1)
                
                y_loss = -torch.mean(torch.log(torch.clip(e_logphi,min=1e-5))*sub_y+torch.log(torch.clip(logphi_o-e_logphi,min=1e-5))*(1-sub_y))

                # calculate MF losss
                inv_prop = 1/torch.clip(self.sigmoid(self.propensity_model_2.forward(sub_x)).detach(), gamma, 1)
                pred = self.sigmoid(g_y)
                imputation_y = self.sigmoid(self.imputation_model.predict(sub_x)).cuda()

                pred_u = self.sigmoid(self.prediction_model.forward(x_sampled) )
                imputation_y1 = self.sigmoid(self.imputation_model.predict(x_sampled)).cuda()

                xent_loss = torch.sum(F.mse_loss(pred.reshape(-1), sub_y.reshape(-1), reduction="none")*inv_prop.reshape(-1)) # o*eui/pui
                imputation_loss = F.mse_loss(pred.reshape(-1), imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss)
                direct_loss = F.mse_loss(pred_u, imputation_y1, reduction="sum")

                mf_loss = (ips_loss + direct_loss)/x_sampled.shape[0]


                # phi_gy=self.normal.cdf(g_y)
                # mf_loss=F.binary_cross_entropy_with_logits(g_y.reshape(-1),sub_y.reshape(-1))
                if alpha is None:
                    y_loss = y_loss*torch.clip(self.rho().detach(),min=0.01,max=0.99)+torch.clip(1-self.rho().detach(),min=0.01,max=0.99)*mf_loss
                else:
                    y_loss = y_loss*alpha+(1-alpha)*mf_loss


                optimizer.zero_grad()
                y_loss.backward()
                optimizer.step()

                # update imputation model
                pred = self.sigmoid(self.prediction_model.predict(sub_x)).cuda()
                imputation_y = self.sigmoid(self.imputation_model.forward(sub_x)).cuda()

                
                e_loss = F.mse_loss(pred, sub_y, reduction="none")
                e_hat_loss = F.mse_loss(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step() 

                # update g_o

                g_y = self.prediction_model.forward(x_sampled_o1, False).unsqueeze(-1).detach()
                g_o = self.propensity_model.forward(x_sampled, False).unsqueeze(-1)

                sub_epsilon = epsilon_all[x_sampled_idx][sub_o==1]

                pos_idx = sub_epsilon + g_y > 0

                logphi=self.normal.cdf((g_o[sub_o==1]+self.rho()*sub_epsilon)/torch.sqrt(1-self.rho()**2)) * pos_idx.detach()
                e_logphi=torch.mean(logphi,axis=1)
                logphi_o = self.normal.cdf(g_o[sub_o==1]).squeeze(-1)
                # logphi_o0=self.normal.cdf(-g_o[idx_o0])
                
                o_loss1 = -torch.sum(torch.log(torch.clip(e_logphi,min=1e-5))*y_sampled+torch.log(torch.clip(logphi_o-e_logphi,min=1e-5))*(1-y_sampled))
                o_loss0=-torch.sum(torch.clip(self.normal.cdf(-g_o[sub_o==0]),min=1e-5).log())
                o_loss=o_loss0+o_loss1
                optimizer.zero_grad()
                o_loss.backward()
                optimizer.step()

                

                xent_loss = y_loss+o_loss



                epoch_loss += xent_loss.detach().cpu().numpy()


            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    # print("[MF_BV] epoch:{}, xent_o1:{}, xent_o0:{}, xent_o1_y:{}, xent_o1_phi:{}".format(epoch, epoch_loss_o1, epoch_loss_o0,epoch_loss_o1_y,epoch_loss_o1_phi))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 1 == 0 and verbose:
                print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                # print("[MF_BV] epoch:{}, xent_o1:{}, xent_o0:{}".format(epoch, epoch_loss_o1, epoch_loss_o0))

            if epoch == num_epoch - 1:
                print("[MF_BV] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def fit_no_o(self, x, y, batch_size=2048,
            num_epoch=1000, lr=0.001, lamb=0,
            tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        

        last_loss = 1e9

        num_sample = len(x)


        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size *
                                       idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx]).cuda()

                g_y, _, _ = self.prediction_model.forward(sub_x, True)

                # log_o1=-((sub_y-g_y)**2)/2

                xent_loss = self.xent_func(g_y,sub_y)
                # xent_loss=nn.MSELoss()(g_y,sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                # print(self.rho.grad)
                optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 1 == 0 and verbose:
                print("[MF_BV] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_BV] Reach preset epochs, it seems does not converge.")
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=1e-4, 
        tol=1e-5, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model_2.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = self.x_all
        early_stop = 0


        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model_2.forward(x_sampled)
                prop = self.sigmoid(prop)
                # propensity score

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")