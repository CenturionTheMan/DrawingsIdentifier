# Introduction

![](.github/readme_img/mainImg.png?raw=true)

The DrawingsIdentifier project is a desktop application designed to
classify user-drawn sketches,
assigning them to one of nine classes contained in the
[QuickDrawDataset](https://github.com/googlecreativelab/quickdraw-dataset).

The utilized classes are:

- axe
- cactus
- cat
- diamond
- moustache
- pants
- snowman
- stairs
- sword

The application utilizes the proprietary [ToyNeuralNetwork](https://github.com/CenturionTheMan/ToyNeuralNetwork)
project to manage feed-forward and convolutional neural networks.
DrawingsIdentifier enables the user to create, train, and test neural networks for data classification tasks.

# User guide

### Sketching window

![](.github/readme_img/page1.1.png?raw=true)

The above figure shows the window that appears to the
user upon launching the application - this is the sketching window.
Using it, the user can create drawings that are
classified in real time into one of nine categories by
two neural networks.

### Training window

![](.github/readme_img/page2.1.png?raw=true)

The next item in the menu is the training window. In this section, the user can:

- view the network's performance on the test and training datasets,
- save and load the network from a file,
- conduct network training.

The last functionality requires prior loading of training data, which can be done in a different window. Attempting to start training without the mentioned data will prompt the user with an appropriate message.

![](.github/readme_img/page2.2.png?raw=true)

The above picture shows the appearance of the application while
training one of the networks.
The training progress can be determined based on two progress bars.
The higher one indicates what percentage of samples in the epoch has already been processed.
The second shows which epoch is currently being processed.
Additionally, the average error of the last batch is displayed in real time.
It is worth noting that the application supports training two networks simultaneously,
but switching between views is blocked as long as at least one of the networks is in
the training process.

![](.github/readme_img/page2.3.png?raw=true)

The figure illustrates the appearance of the screen after stopping the training. It shows that the statistical performance of the network has been calculated for both the training and test datasets.

### Configuration window

![](.github/readme_img/page3.1.png?raw=true)

The next two screens, marked with Roman numerals I and II, are used for configuring the structure and learning parameters of the network (following figure shows the view of screen I).  
In this window, we can distinguish two groups. The upper one is used to configure the parameter values of the Trainer class, which is used for training the network in the window described in the previous chapter. Changes made in this panel are saved in real time.  
The second group is used for modeling the network structure. In it, we can specify how many layers the model should contain and, depending on the layer type, set its parameters. Applying changes in this section requires manually pressing the save button. This is because ering the network structure results in the loss of training progress.

### Data loading window

![](.github/readme_img/page4.1.png?raw=true)

Data loading window allows the user to load sample data from the "Quick Draw Dataset"
into the application. The user can select the number of drawings from a given class
to load and can start the data loading process after specifying the path to the folder
containing the .npy files.

![](.github/readme_img/page4.2.png?raw=true)

Once the data loading is complete, the screen will be filled with randomly selected
samples from the loaded files, as shown in above picture. Based on these,
the user can determine what type of images the application should expect.

### User guide window

![images](.github/readme_img/page5.1.png?raw=true)

Following figure shows the last window containing the user guide.
It provides a condensed description of each of the application's screens.
