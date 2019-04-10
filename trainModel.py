from __future__ import division
import time
import torch
import os
import random
from scipy import misc
import numpy as np
import genSamps_3 as gs

HOW_MANY_TO_USE = 10000
H1 = 5000
H2 = 1000
learning_rate = 1e-6
EPOCHS = 5000
PRINT_EVERY = 5

MODEL_PATH = 'myModelBiggest.pt'


outdir = 'out/'+str(time.time())+"/"

os.system("mkdir "+outdir)
os.system("cp trainModel.py "+outdir)

print("Loading X")
resX = gs.gen("x",HOW_MANY_TO_USE)

print("Loading Y")
resY = gs.gen("y",HOW_MANY_TO_USE)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = resX.shape[0]
D_in = 13068 # resX.shape[1]     (it is 33*33*3*4)
D_out = 3267 # resY.shape[1]	 (it is 33*33*3 )

print(N)
print(D_in)
print(D_out)

print("Making tensors out of numpy")
# Create random Tensors to hold inputs and outputs
x = torch.tensor(resX).float()
y = torch.tensor(resY).float()

print(x)
print(y)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(H2, H2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(H2, H2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(H2, H2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(H2, H2),
    torch.nn.ReLU(),
    # torch.nn.Dropout(0.1),
    torch.nn.Linear(H2, D_out),
)


#model.load_state_dict(torch.load(MODEL_PATH))





# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

for t in range(EPOCHS):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # if t > 10:
    #     learning_rate = 5e-7

    if t % PRINT_EVERY == 0:
        oneToUse = random.randint(0,N-1)
        gs.revy(y_pred.data.numpy().copy()[oneToUse],t,oneToUse,'samp',outdir)
        torch.save(model.state_dict(), outdir+MODEL_PATH)
        #gs.revy(y.data.numpy().copy()[oneToUse],t,oneToUse,'actuy')
        #gs.revx(x.data.numpy().copy()[oneToUse],t,oneToUse,'actux')

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad








