This project is a continuation of the [PIHNN project](https://github.com/teocala/pihnn) by Matteo Calafà.

The extensions are work in progress by Nicolas Cuenca, Jonas Hund, and Tito Andriollo.

Questions and Remarks:
- I do not understand the script to run the cracked plate test. In my environment, the test failed because mismatched dimensions of target values and computed values: The number of given data point coordinates has to match the number of precribed stress values.
- Accordingly, I changed some functions arguments'. The x- and y-coordinates of the points where the stress is evaluated now have to be passed on as lists to the "train_devo_adam" method as the calculation of "data_loss" requires the coordinates.
- In which order have the scripts to be run to yield meaningful results and what are the corresponding parameters? The target values for the stresses are currently manually added in, I assume?
- Are the target values extracted from Abaqus simulations at this stage? 
