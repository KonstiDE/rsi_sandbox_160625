import torch
import config.config as cfg

from tqdm import tqdm

from model.rsi_model import RSIModel
from provider.rsi_provider import get_loader

def test():
    saved_model = torch.load("models/rsi_model_10.pt")
    model = RSIModel()
    model.load_state_dict(saved_model)

    test_loader = get_loader(cfg.get_test_path())

    loop = tqdm(test_loader)

    correctly_predicted = 0

    for (data, target) in loop:
        pred = model(data)

        class_predicted = torch.argmax(pred, dim=1)

        if class_predicted == target:
            correctly_predicted += 1

    print("Accuracy {}".format(correctly_predicted / len(test_loader.dataset)))


if __name__ == '__main__':
    test()