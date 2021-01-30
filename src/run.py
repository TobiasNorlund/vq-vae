import torch


def run_one_vae_epoch(model, data_loader, optimizer, is_training: bool):

    if is_training is True:
        optimizer.zero_grad()
    tot_loss = 0
    n = 0
    kldivloss = 0
    recloss = 0
    for batch, labels in data_loader:
        (mu, logsigma), recons = model(batch)

        if is_training is True:
            optimizer.zero_grad()

        # Now, compute the loss
        loss, loss_dict = model.loss(mu, logsigma, x=batch, x_hat=recons)
        tot_loss += loss
        kldivloss += loss_dict["kldivloss"]
        recloss += loss_dict["recloss"]

        if is_training is True:
            loss.backward()
            optimizer.step()
        n += 1
    tot_loss /= n
    kldivloss /= n
    recloss /= n
    return tot_loss, recloss, kldivloss


def train_vae(model, n_epochs: int, train_loader, val_loader, optimizer):
    if torch.cuda.is_available() is True:
        model.cuda()

    for epoch in range(n_epochs):

        losses = run_one_vae_epoch(
            model, data_loader=train_loader, optimizer=optimizer, is_training=True
        )

        print(
            "Train losses: \nTot: {}\nKLDivLoss: {}\nRec loss: {}".format(
                losses[0], losses[1], losses[2]
            )
        )
        losses = run_one_vae_epoch(
            model, data_loader=val_loader, optimizer=None, is_training=False
        )
        print(
            "Validation losses: \nTot: {}\nKLDivLoss: {}\nRec loss: {}".format(
                losses[0], losses[1], losses[2]
            )
        )
    return model


def eval(model, test_loader):
    pass
