#pragma once

#include <stdio.h>

namespace ztool
{

typedef void(*PrintfCallbackFunc)(const char*, va_list);

PrintfCallbackFunc setPrintfCallback(PrintfCallbackFunc func);

void lprintf(const char* format, ...);

}