import torch


class Neumf(torch.nn.Module):
    def __init__(self, config):
        super(Neumf, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        # print(self.embedding_user_mlp) # (597, 8)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        # self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # print(user_indices.size())  # torch.Size([128=batch_size])
        # print(user_indices)
        # print(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        # print(user_embedding_mlp)
        # print(user_embedding_mlp.size()) # torch.Size([128=batch_size, 8])
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        # print(item_embedding_mlp)
        # print(item_embedding_mlp.size())
        user_embedding_mf = self.embedding_user_mf(user_indices)
        # print(user_embedding_mf)
        # print(user_embedding_mf.size())
        item_embedding_mf = self.embedding_item_mf(item_indices)
        # print(item_embedding_mf)
        # print(item_embedding_mf.size())
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        # print(mlp_vector)
        # print(mlp_vector.size()) # torch.Size([128=batch_size, 8+8=16])
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        # print(mf_vector)
        # print(mf_vector.size()) # torch.Size([128=batch_size, 8=8])

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # print(mlp_vector)
        # print(mlp_vector.size())

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        # print(vector)
        # print(vector.size())
        # logits = self.affine_output(vector)
        # rating = self.logistic(logits)
        # return rating
        return vector