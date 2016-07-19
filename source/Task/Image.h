#pragma once

const int watermarkWidth = 256;
const int watermarkHeight = 128;

extern unsigned char* watermarkData;

void setWatermarkLanguage(bool isChinese);

extern int addWatermark;