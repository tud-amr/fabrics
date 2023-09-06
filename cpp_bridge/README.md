# Export fabrics as c code

Casadi allows to export functions as native c code without any dependencies.
In this package, we have implemented a way to export a composed planner using
the function `planner.export_as_c(file_name)`. This function generates a single
c-file that can be used in a simulation loop.

Here we give a very brief version of how to use it. Move your exported
`planner.c` file into the `src`-folder.

Then, create a `build`-directory and run:
```bash
cd build
cmake ..
make 
./main
```

Potential Segmentations faults are usually caused by the wrong number of inputs
provided to the function `casadi_f0` defined in `planner.c`.


Good luck with that.
For questions, don't hesitate to create an issue.
