import pickle
import torch

f = open('cnn.pkl', 'rb')
model = pickle.load(f)
f.close()
print(type(model))
# model.load_state_dict(torch.load('cnn.pth'))
