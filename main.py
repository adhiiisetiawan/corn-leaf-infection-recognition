from src.dataset import dataset, dataloader


if __name__ == "__main__":
    train_set, val_set, test_set = dataset(data_path='./data/processed/',
                                           crop_size=224, 
                                           training_size=0.75, 
                                           validation_size=0.15)

    trainloader, validationloader, testloader = dataloader(train_set, 
                                                           val_set, 
                                                           test_set, 
                                                           batch_size=128)


