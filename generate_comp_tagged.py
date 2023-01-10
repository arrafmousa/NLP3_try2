from finalModel import *


def generate_comp_tagged(model_name=r'night_model_after_4_epoch'):
    model_ = DependencyParser(100).to(device)  # TODO :  change the embedding dim from 100 (200/250 for example)

    model_.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    comp_f = open("comp_212701239_207571258.labeled", "w")
    f = open(r'comp.unlabeled')
    file = f.read().split("\n\n")
    for sen in file:
        if len(sen) == 0:
            continue
        df = pd.DataFrame([row.split('\t') for row in sen.split('\n')],
                          columns=["Token Counter", "Token", "-1", "Token POS", "-2", "-3", "Token Head",
                                   "Dependency Label", "-4", "-5"])
        sentence = df[["Token", "Token POS", "Token Head"]]
        _, T = model_(sentence)
        Token_heads = T[0][1:]

        Tokens = np.array([x for x in sentence["Token"].values])
        Token_POSes = np.array([x for x in sentence["Token POS"].values])
        for i in range(len(Tokens)):
            comp_f.write(str(i+1) + '\t' + Tokens[i] + '\t' + '_' + '\t' + Token_POSes[i] + '\t' + '_' + '\t' + '_' +
                         '\t' + str(Token_heads[i]) + '\t' + '_' + '\t' + '_' +
                         '\t' + '_' + '\n')
        comp_f.write('\n')


if __name__ == "__main__":
    generate_comp_tagged()
    print("DONE!")
