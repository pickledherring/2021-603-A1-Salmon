## Serial, multithreaded, and MPI implementations of a KNN
---
`main` is the serial version, `main_par` is the pthread parallelized version, and `main_mpi` will be the mpi version.

Run the code with `./<version> datasets/<size>-train datasets/<size>-test <K> <#threads>`.\
Leave off `#threads` if running `main`.

For example:\
`./main_par datasets/small-train datasets/small-test 3 8`