#pragma once

#define OPENCL_KERNEL_STRING( ... )# __VA_ARGS__

#define OPENCL_COMPILE_DEV 1

#if OPENCL_COMPILE_DEV
#define APPEND_BUILD_OPTION " -D DEV"
#define PROG_FILE_NAME(x) x
#define PROG_STRING(x) ""
#else
#define APPEND_BUILD_OPTION ""
#define PROG_FILE_NAME(x) L""
#define PROG_STRING(x) x
#endif