import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(hid_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.linear(x)
        return ret


def node_evaluation(emb, y, train_idx, valid_idx, test_idx, epochs=200, lr=1e-2, weight_decay=1e-4):
    device = emb.device

    nclass = y.max().item() + 1
    logreg = LogReg(emb.shape[1], nclass).to(device)
    train_idx, valid_idx, test_idx, y = train_idx.to(device), valid_idx.to(device), test_idx.to(device), y.to(device)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0
    pred = None

    for epoch in range(epochs):
        logreg.train()
        opt.zero_grad()

        logits = logreg(emb)

        preds = torch.argmax(logits[train_idx], dim=1)
        train_acc = torch.sum(preds == y[train_idx]).float() / train_idx.size(0)

        loss = loss_fn(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()


        logreg.eval()
        with torch.no_grad():

            if valid_idx.size(0) != 0:
                val_logits = logreg(emb[valid_idx])
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = torch.sum(val_preds == y[valid_idx]).float() / valid_idx.size(0)
            else:
                train_logits = logreg(emb[train_idx])
                train_preds = torch.argmax(train_logits, dim=1)
                train_acc = torch.sum(train_preds == y[train_idx]).float() / train_idx.size(0)
                val_acc = train_acc
            
            test_logits = logreg(emb[test_idx])
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = torch.sum(test_preds == y[test_idx]).float() / test_idx.size(0)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc
                    pred = test_preds

    return eval_acc, pred
