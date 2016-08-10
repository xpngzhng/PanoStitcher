#pragma once

#define OPENCL_KERNEL_STRING( ... )# __VA_ARGS__

// The following macro determins how OpenCL source code is compiled.
// If it is set to 0, source code is read from const char * strings in cpp files,
// else source code is read from external text files, and 
// we should make sure the text file exists in the place where the exe is.
#define OPENCL_COMPILE_DEV 0

#if OPENCL_COMPILE_DEV
#define PROG_FILE_NAME(x) x
#define PROG_STRING(x) ""
#else
#define PROG_FILE_NAME(x) L""
#define PROG_STRING(x) x
#endif