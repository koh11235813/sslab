# federated/client.py
import torch
import flwr as fl
from task import load_task   # task名から(model, loss, metrics, dataloaders)を返す
from semantic.encoder import quantize_8bit, sparsify_topk

class JetsonClient(fl.client.NumPyClient):
    def __init__(self, cfg):
        self.model, self.crit, self.metrics, self.loaders = load_task(cfg)
        self.cfg = cfg

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def fit(self, parameters, config):
        # 受信重みをロード
        sd = self.model.state_dict()
        for (k, v), npv in zip(sd.items(), parameters):
            sd[k] = torch.tensor(npv).to(sd[k].device).type_as(sd[k])
        self.model.load_state_dict(sd, strict=False)

        # ローカル学習
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        for _ in range(self.cfg.local_epochs):
            for x, y in self.loaders["train"]:
                pred = self.model(x.to(self.cfg.device))
                loss = self.crit(pred, y.to(self.cfg.device))
                opt.zero_grad(); loss.backward()
                if self.cfg.semantic == "gradients_topk":
                    for p in self.model.parameters():
                        if p.grad is None: continue
                        p.grad.data = sparsify_topk(p.grad.data, k=self.cfg.topk)
                opt.step()

        # 送信用（重み or 差分を量子化）
        params = [p.detach().cpu() for p in self.model.state_dict().values()]
        if self.cfg.semantic == "features_8bit":
            params = [quantize_8bit(p).cpu().numpy() for p in params]
        else:
            params = [p.numpy() for p in params]
        return params, len(self.loaders["train"].dataset), {}

    def evaluate(self, parameters, config):
        # 同様に重みを適用して評価
        return 0.0, 0, {}

def main(cfg):
    fl.client.start_numpy_client(server_address=cfg.server, client=JetsonClient(cfg))

