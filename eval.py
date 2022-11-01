import torch
import pickle
import numpy as np
import logging
from newmodel import NewModel

class Eval(object):
    def __init__(self, data_loader, args, max_len, n_links):
        self.model = NewModel(args, max_len, n_links)
        self.device = args.device
        self.model.to(self.device)
        self.data_loader = data_loader
        self.model_dir = args.model_dir
        self.data_list = None

        if args.eval_file:
            self.result_file = f".{args.model_dir.split('.')[1]}_{args.eval_file.split('_')[2]}_result.pt"
        else:
            self.result_file = f".{args.model_dir.split('.')[1]}_result.pt"

        self.logger = logging.getLogger(args.mode)
        
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.mae = torch.nn.L1Loss()

    # 皮尔森系数计算
    def pearsonr(self, y_pred, y_true):
        y_pred, y_true = y_pred.view(-1), y_true.view(-1)
        centered_pred = y_pred - torch.mean(y_pred)
        centered_true = y_true - torch.mean(y_true)
        covariance = torch.sum(centered_pred * centered_true)
        bessel_corrected_covariance = covariance / (y_pred.size(0) - 1)
        std_pred = torch.std(y_pred, dim=0)
        std_true = torch.std(y_true, dim=0)
        corr = bessel_corrected_covariance / (std_pred * std_true)
        return corr

    def eval(self):
        self.load_model()
        self.logger.info("# evaluating model performance ...")

        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for bw, tr, p_lidx, mask, y in self.data_loader:
                bw, tr, p_lidx, mask = bw.to(self.device), tr.to(self.device), p_lidx.to(self.device), mask.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(bw, tr, p_lidx, mask)
                y_pred.append(y_hat.view(-1).to('cpu'))
                y_true.append(y.view(-1).to('cpu'))
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            mse = self.mse(y_pred, y_true)
            mae = self.mae(y_pred, y_true)
            rho = self.pearsonr(y_pred, y_true)
            mre = torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true))
        log = "MSE: %.6f | MAE: %.5f | RHO: %.5f | MRE: %.5f" % (mse, mae, rho, mre)
        self.logger.info(log)
        self.logger.info(f"# saving test results in {self.result_file}")
        with open(self.result_file, 'wb') as f:
            pickle.dump((y_pred, y_true), f)
    
    # def set_dataloader(self, dataloader):
    #     self.data_loader = dataloader

    def load_model(self):
        self.logger.info("# loading model weight ...")
        best_model = torch.load(self.model_dir)
        self.model.load_state_dict(best_model)

    # def eval_by_traffic(self, data_list):
    #     self.load_model()
    #     self.logger.info("# evaluating model performance ...")

    #     self.model.eval()
    #     y_pred = []
    #     y_true = []
    #     with torch.no_grad():
    #         for data_name, dataloader in data_list.items():
    #             for bw, tr, p_lidx, mask, y in dataloader:
    #                 bw, tr, p_lidx, mask =  bw.to(self.device), tr.to(self.device), p_lidx.to(self.device), mask.to(self.device)
    #                 y = y.to(self.device)
    #                 y_hat = self.model(bw, tr, p_lidx, mask)
    #                 y_pred.append(y_hat.view(-1).to('cpu'))
    #                 y_true.append(y.view(-1).to('cpu'))
    #             y_pred = torch.cat(y_pred, dim=0)
    #             y_true = torch.cat(y_true, dim=0)
    #             mse = self.mse(y_pred, y_true)
    #             mae = self.mae(y_pred, y_true)
    #             rho = self.pearsonr(y_pred, y_true)
    #             mre = torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true))
    #             log = "# %s : MSE: %.5f | MAE: %.5f | RHO: %.5f | MRE: %.5f" % (data_name, mse, mae, rho, mre)
    #             self.logger.info(log)
    #             traffic_intensity = int(data_name.split('_')[2])




    
    # def save_data(self):
    #     self.logger.info("# loading model weight ...")
    #     best_model = torch.load(self.args.model_dir)
    #     self.model.load_state_dict(best_model)
    #     self.logger.info("# computing target results ...")

    #     self.model.eval()
    #     data_y = []
    #     data_y_hat = []
    #     with torch.no_grad():
    #         for bw, tr, p_lidx, mask, y in self.data_loader:
    #             bw, tr, p_lidx, mask = bw.to(self.device), tr.to(self.device), p_lidx.to(self.device), mask.to(self.device)
    #             batch = bw.size(0)
    #             y = y.to(self.device)
    #             y_hat = self.model(bw, tr, p_lidx, mask)
    #             data_y.append(y.view(batch, -1).to('cpu'))
    #             data_y_hat.append(y_hat.view(batch, -1).to('cpu'))
            
    #     self.logger.info("# saving results in csv file ...")
    #     data_y = torch.cat(data_y, dim=0)
    #     data_y_hat = torch.cat(data_y_hat, dim=0)
    #     np.savetxt('./data_y.csv', data_y.detach().numpy(), fmt='%f', delimiter=',')
    #     np.savetxt('./data_y_hat.csv', data_y_hat.detach().numpy(), fmt='%f', delimiter=',')