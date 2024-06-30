import torch


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size=400,
        h2_size=300,
        lr=1e-3,
        chkpt_path="weights/actor.pt",
    ):
        super(ActorNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = lr
        self.chkpt_path = chkpt_path

        self.h1_layer = torch.nn.Linear(*self.input_shape, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)

        self.out_layer = torch.nn.Linear(self.h2_size, *self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.h1_layer(state))
        x = torch.nn.functional.relu(self.h2_layer(x))
        return torch.nn.functional.tanh(self.out_layer(x))

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
