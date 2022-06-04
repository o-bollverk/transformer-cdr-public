import torch
from torch.utils.data import DataLoader

def train_epoch_dataloader(model, cdr_dataset, device, batchsize, optimizer, loss_fn):
    model.train()
    losses = 0
    train_dataloader = DataLoader(cdr_dataset, batch_size=batchsize)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        if torch.is_floating_point(src): # if input is x y coordinate, it is float, if array of cell-IDs, it's not, and no gradient can be set
            src.requires_grad = True

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_input)
        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def predict(model, cdr_dataset, device, batchsize, loss_fn, return_losses = False):
    model.eval()
    losses = 0
    dataloader = DataLoader(cdr_dataset, batch_size=batchsize)
    estimated_routes = []
    with torch.no_grad():
        #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/37
        # only useful for saving memory
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]

            logits = model(src, tgt_input)

            estimated_route = torch.argmax(logits,  axis = 2)
            estimated_routes.append(estimated_route)

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            losses += loss.item()

    estimated_routes = torch.cat(estimated_routes, axis = 0)

    if return_losses:
        return estimated_routes, losses
    else:
        return estimated_routes
