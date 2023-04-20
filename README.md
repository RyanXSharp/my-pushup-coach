# My Pushup Coach
A web app that uses machine learning to critique your form while you do push-ups. https://my-pushup-coach.herokuapp.com/

More specifically, the model uses your webcam to:
 - Detect if you are in push-up position
 - Detect if your bottom is too high, too low, or just right.
 - Detect if your head is hanging too low or is up.
 - Measure the depth of your push-up, which enables the model to count your push-ups and detect if you didn't go down deep enough.

The "Data" folder contains all data that my models were trained on.

The "my pushup coach" folder contains the web app that was deployed.

The "training the models.py" Python file contains the code which was used to collect the training data and train the Ml models.
