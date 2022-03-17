from tqdm.auto import tqdm

def loop_function(mode, dataset, dataloader, model, criterion, optimizer, device):
    """_summary_

    Args:
        mode (_type_): _description_
        dataset (_type_): _description_
        dataloader (_type_): _description_
        model (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    if mode == "train":
        model.train()
    elif mode == "val":
        model.eval()

    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)

        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc