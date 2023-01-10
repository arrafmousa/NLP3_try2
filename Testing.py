from finalModel import *


def train(model_name=r'night_model_after_4_epoch'):
    testing_set = get_df(r'test.labeled')
    model_ = DependencyParser(100).to(device)  # TODO :  change the embedding dim from 100 (200/250 for example)

    model_.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    accuracies = []
    for idx, sentence in tqdm(enumerate(testing_set)):
        loss, T = model_(sentence)
        accuracy = np.sum(T[0][1:] == np.array([int(x) for x in sentence["Token Head"].values])) / T[0].size
        accuracies.append(accuracy)

    total_acc = sum(accuracies) / len(accuracies)
    if total_acc > 0.7:
        print(f"V-V-V-V-V accuracy \t {total_acc}")
    else:
        print(f"*X*X*X*X accuracy \tonly got {total_acc}")


if __name__ == "__main__":
    train()
