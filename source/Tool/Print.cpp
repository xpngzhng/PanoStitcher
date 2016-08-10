#include "Print.h"
#include <stdarg.h>

namespace ztool
{

void printfDefaultCallback(const char* format, va_list vl)
{
    vprintf(format, vl);
}

PrintfCallbackFunc printfCallback = printfDefaultCallback;

void lprintf(const char* format, ...)
{
    if (printfCallback)
    {
        va_list vl;
        va_start(vl, format);
        printfCallback(format, vl);
        va_end(vl);
    }
}

PrintfCallbackFunc setPrintfCallback(PrintfCallbackFunc func)
{
    PrintfCallbackFunc oldFunc = printfCallback;
    printfCallback = func;
    return oldFunc;
}

}