import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size, emoji_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

def generate_encoding(values):
    encoding_length = len(values)
    if encoding_length == 10:
        encoding_length = 7 # for emojis looool
    return [[1.0 if i == j else 0.0 for i in range(encoding_length)] for j in range(encoding_length)]

char_encodings = generate_encoding(' hatrcflmsonp')
emoji_encodings = generate_encoding('🎩🐀🐈🏢👱‍♂️🧢👦')

encoding_size = len(char_encodings)

index_to_emoji = ['🎩','🐀','🐈','🏢','👱‍♂️','🧢','👦']

def charToIndex(char):
    char_to_number_map = {
        ' ': 0, 'h': 1, 'a': 2, 't': 3,
        'r': 4, 'c': 5, 'f': 6, 'l': 7,
        'm': 8, 's': 9, 'o': 10, 'n': 11,
        'p': 12
    }
    return char_to_number_map.get(char)

def wordTolist(word):
    list = []
    for char in word:
        index = charToIndex(char)
        list.append([char_encodings[index]])
    return list

x_train = torch.tensor([wordTolist('hat '), wordTolist('rat '), wordTolist('cat '), wordTolist('flat'), wordTolist('matt'), wordTolist('cap '), wordTolist('son ')])

y_train = torch.tensor([
    [emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]], [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]], [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],
    [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],
    [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],
    [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],
    [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]],
])

print("x_train.shape ", x_train.shape)
print("y_train.shape ", y_train.shape)

print(y_train)

model = LongShortTermMemoryModel(encoding_size, 7)
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for batch in range(7):
        model.reset()
        model.loss(x_train[batch], y_train[batch]).backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 10 == 9:
            model.reset()
            text = 'matt'
            model.f(torch.tensor(wordTolist(text)))
            y = model.f(torch.tensor(wordTolist(text)))
            text += index_to_emoji[y.argmax(1)[0].item()]

            print(text)
