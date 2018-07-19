# cartpole-keras

We are using a openai environment named carpole to ilustrate the basics of ai.
The Python Deep Learning library used is Keras, which is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. 

--------//-----------//-----------//------------//

First Step - Create a random initial population

In order to create an initial population, we have to make random movements and observe the modification of the environment. Then, we
select the mediocre ones and save what they do when they're seeing the environment.

--------//-----------//-----------//------------//

Second Step - Create a neural netork

Using Keras, we create a neural network, which first layer consist on the input, 5 hidden layers -> 128 - 256 - 512 - 256 - 128 nodes each. The final layer is the output, which gives us the next movement.

--------//-----------//-----------//------------//

Third Step - Train

Using the data of the first step, we train the neural network in order to improve the connections for futures tests.

--------//-----------//-----------//------------//

Forth Step - Repeat until it does 50000000 points

A new population is created based on the results of the tests of the neural network, and then the third step is repeated, now using the 
new data.
