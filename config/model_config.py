class model_config():
    def __init__(self, KG=False, model_name='DNN'):
        if KG:
            self.drug_hidden_dim = [100, 100, 64]
            self.protein_hidden_dim = [100, 100, 64]
        else:
            self.drug_hidden_dim = [2048, 1024, 256, 128]
            self.protein_hidden_dim = [420, 200, 100, 32]
        self.lr = 0.001
        self.epoch = 500
        self.model_path = 'ckpts/dti_model/{}_best_lr{}.pth'.format(model_name, self.lr)
