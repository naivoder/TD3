import torch


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size=400,
        h2_size=300,
        lr=1e-3,
        decay=1e-2,
        chkpt_path="weights/critic.pt",
    ):
        super(CriticNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = lr
        self.decay = decay
        self.chkpt_path = chkpt_path

        input_features = input_shape[0] + self.n_actions[0]
        self.h1_layer = torch.nn.Linear(input_features, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)

        self.out_layer = torch.nn.Linear(self.h2_size, 1)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.decay
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = torch.concatenate((state, action), dim=1)

        x = torch.nn.functional.relu(self.h1_layer(x))
        x = torch.nn.functional.relu(self.h2_layer(x))

        return self.out_layer(x)

    def save_checkpoint(self, epoch=None, loss=None):
        torch.save(self.state_dict(), self.chkpt_path)
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model_state_dict": self.state_dict(),
        #         "optimizer_state_dict": self.optimizer.state_dict(),
        #         "loss": loss,
        #     },
        #     self.chkpt_path,
        # )

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_path))
        # chkpt = torch.load(self.chkpt_path)
        # self.load_state_dict(chkpt["model_state_dict"])
        # self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        # epoch = chkpt["epoch"]
        # loss = chkpt["loss"]
        # return epoch, loss
