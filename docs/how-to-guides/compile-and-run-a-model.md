# How to compile and run a model

You have an FFSL model file and the input data it needs.
You want a simulator binary and an output file.

This guide assumes `ffsc` is installed,
and that Conan and CMake are on your path.

## 1. Compile the model to a C++ project

```
ffsc compile -i examples/example1.ffsl -o build/sim
```

The command prints nothing when it succeeds.
It writes three files into the output directory.

* `simulator.cpp` - the generated simulator.
* `CMakeLists.txt` - the CMake build definition.
* `conanfile.py` - the Conan recipe listing the C++ dependencies.

If the compiler rejects your model,
it prints a single located message and exits with status 1.
A syntax error is the exception.
It prints a Python traceback,
and the last line of the traceback gives the line and column.
It also exits with status 1.
Fix the reported line.
Then run the command again.

## 2. Build the simulator

```
cd build/sim
conan build . --build=missing
```

The generated `CMakeLists.txt` asks for C++23.
If your default Conan profile sets an older standard,
set the standard on the command line instead,
or the configure step fails.

```
conan build . --build=missing -s compiler.cppstd=gnu23
```

With a single-configuration generator,
find the executable at `build/<build_type>/simulator`
underneath the project directory,
for example `build/Release/simulator`.
The path has this shape because the recipe uses `cmake_layout`.

If you want a build type other than `Release`,
pass it to the build.
Then use it in the path you run afterward.

```
conan build . --build=missing -s build_type=Debug
```

## 3. Run the simulator

Pass the grid size and the tick count as flags.
The simulator does not read them from the input files,
so make them agree with the files you pass.

```
build/Release/simulator \
    --num-ticks 100 \
    --num-rows 4096 \
    --num-cols 4096 \
    --tile-file tiles.h5 \
    --seed-file seed.h5 \
    --tick-file ticks.parquet \
    --output-file output.h5
```

The full flag list is in
[The simulator command line](../reference/simulator-command-line.md).
The file layouts are in
[Input and output files](../reference/input-and-output-files.md).

## Change a parameter without recompiling

Every `config` declared in the FFSL source is a command line flag.
If you want to sweep `base_ember_rate`,
run the same binary with a different value each time.

```
for rate in 0.5 0.7 0.9; do
    build/Release/simulator \
        --num-ticks 100 --num-rows 4096 --num-cols 4096 \
        --tile-file tiles.h5 --seed-file seed.h5 --tick-file ticks.parquet \
        --output-file "output-$rate.h5" \
        --config-base-ember-rate "$rate"
done
```

Anything declared as an `option` rather than a `config`
is fixed at compile time.
If you change one, go back to step 1.

## Set the thread count

Set the thread count in the environment.
The simulator has no flag for it.

```
OMP_NUM_THREADS=8 build/Release/simulator ...
```

If you need a run you can repeat exactly, use one thread.
Several threads do not repeat, even from the same seed.
[Reproducibility](../explanation/reproducibility.md) explains why.

## Repeat an earlier run

If you did not record the seed,
read it back from the output file of that run.

```
python3 -c "import h5py; print(h5py.File('output.h5')['runstats'].attrs['seed'])"
```

Then pass it to a single-threaded run.

```
OMP_NUM_THREADS=1 build/Release/simulator ... --seed 42
```
