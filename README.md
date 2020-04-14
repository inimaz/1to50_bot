# Game bot - 1to50
Bot to play the 1to50 (zzzscore.com/1to50/) online game.

Note: this is a fork from [imohitmayank](https://github.com/imohitmayank/1to50_bot). Thumbs up to him for the great initial work.

I added:
* save the screenshot of the image location
* give human feedback in case the neural network (NN) missed any number

### Usage

To load the image and train the neural network

```
from game_bot_1to50 import *
model = train_network(image_directory = "imgs")
save_model(model, "1to50_trained_model.h5")
```
To load the model and play the game

```
model = load_model("1to50_trained_model.h5")
# Define the co-ordinate of the gme board
image_location = {'top': 239, 'left': 486, 'width': 380, 'height': 378}
play_game(image_location)

```

## See image location
Note that the image location is the exact position of the board on your screen.
How to know if the position is right? By trial and error, taking screenshots.
Use the function save_screenshot to verify that the location is correct.
```
save_screenshot (image_location)
```


## Optimization
Sometimes the NN does not read all numbers correctly. Do you still want to have fun using it?
Help it by running the find_and_click function instead:


    model = load_model("new.h5")
    # Define variables
    image_location = {'top': 295, 'left': 287, 'width': 380, 'height': 378}
    find_and_click(image_location, model)
    save_screenshot(image_location)
The NN will try to solve it on its own and will ask for help on the numbers she missed. Team effort!