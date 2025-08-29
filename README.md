This project is a continuation of the [PIHNN project](https://github.com/teocala/pihnn) by Matteo Calafà.

The extensions are work in progress by Nicolas Cuenca, Jonas Hund, and Tito Andriollo.

TO DO:
- Class enriched_PIHNN_devo has to inherit from class DD_PIHNN from the nn module of package pihnn.
- I do not understand the script to run the cracked plate test. In my environment, the test fails because mismatched dimensions of target values and computed values.
- The number of given data point coordinates has to match the number of precribed stress values.
- The x- and y-coordinates have to be passed on as lists to the "train_devo_adam" method as the calculation of "data_loss" requires the coordinates of the points at which the loss shall be evaluated. (This interpretation of the current code needs to be checked.) 
