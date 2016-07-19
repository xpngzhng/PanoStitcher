#include "ticpp.h"
#include "ZReproject.h"
#include "Rotation.h"
#include "Mtx33.h"

#include <fstream>
#include <sstream>

PhotoParam::PhotoParam()
{
    memset(this, 0, sizeof(PhotoParam));
}

using ticpp::Element;
using ticpp::Document;

static bool loadPhotoParamFromXML1(const std::string& fileName, std::vector<PhotoParam>& params)
{
    params.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* pFirstRoot = doc.FirstChildElement("Root", false);
    if (pFirstRoot == NULL)
    {
        params.clear();
        return false;
    }

    for (ticpp::Iterator<ticpp::Element> pRoot(pFirstRoot, "Root"); pRoot != pRoot.end(); ++pRoot)
    {
        PhotoParam param;
        memset(&param, 0, sizeof(PhotoParam));

        bool findRectPosInRoot = false;
        double nTemp;

        const Element* pGlobalInfo = pRoot->FirstChildElement("GlobalInfo", false);
        if (pGlobalInfo == NULL)
        {
            params.clear();
            return false;
        }

        Element* pEle = NULL;
        pEle = pGlobalInfo->FirstChildElement("IMAGETYPE", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.imageType = nTemp;

        pEle = pGlobalInfo->FirstChildElement("CROPMODE", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropMode = nTemp;

        pEle = pGlobalInfo->FirstChildElement("HFOV", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.hfov = nTemp;

        pEle = pGlobalInfo->FirstChildElement("VFOV", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.vfov = nTemp;

        pEle = pGlobalInfo->FirstChildElement("CROPLEFT", false);
        if (pEle)
        {
            findRectPosInRoot = true;
            pEle->GetText(&nTemp);
            param.cropX = nTemp;

            pEle = pGlobalInfo->FirstChildElement("CROPRIGHT", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.cropWidth = nTemp - param.cropX;

            pEle = pGlobalInfo->FirstChildElement("CROPTOP", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.cropY = nTemp;

            pEle = pGlobalInfo->FirstChildElement("CROPBOTTOM", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.cropHeight = nTemp - param.cropY;
        }

        pEle = pGlobalInfo->FirstChildElement("Alpha", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.alpha = nTemp;

        pEle = pGlobalInfo->FirstChildElement("Beta", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.beta = nTemp;

        pEle = pGlobalInfo->FirstChildElement("Gamma", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.gamma = nTemp;

        pEle = pGlobalInfo->FirstChildElement("ShiftX", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shiftX = nTemp;

        pEle = pGlobalInfo->FirstChildElement("ShiftY", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shiftY = nTemp;

        pEle = pGlobalInfo->FirstChildElement("ShearX", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shearX = nTemp;

        pEle = pGlobalInfo->FirstChildElement("ShearY", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shearY = nTemp;

        Element* pPos = pRoot->FirstChildElement("POSITION", false);
        Element* pVideo = pRoot->FirstChildElement("VIDEO", false);
        if (pPos == NULL && pVideo == NULL)
        {
            params.clear();
            return false;
        }

        for (ticpp::Iterator<ticpp::Element> child(pPos ? pPos : pVideo, pPos ? "POSITION" : "VIDEO"); child != child.end(); child++)
        {
            const Element* pEle = NULL;

            if (!findRectPosInRoot)
            {
                pEle = child->FirstChildElement("CROPLEFT", false);
                if (pEle == NULL)
                {
                    params.clear();
                    return false;
                }
                pEle->GetText(&nTemp);
                param.cropX = nTemp;

                pEle = child->FirstChildElement("CROPRIGHT", false);
                if (pEle == NULL)
                {
                    params.clear();
                    return false;
                }
                pEle->GetText(&nTemp);
                param.cropWidth = nTemp - param.cropX;

                pEle = child->FirstChildElement("CROPTOP", false);
                if (pEle == NULL)
                {
                    params.clear();
                    return false;
                }
                pEle->GetText(&nTemp);
                param.cropY = nTemp;

                pEle = child->FirstChildElement("CROPBOTTOM", false);
                if (pEle == NULL)
                {
                    params.clear();
                    return false;
                }
                pEle->GetText(&nTemp);
                param.cropHeight = nTemp - param.cropY;
            }

            pEle = child->FirstChildElement("YAW", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.yaw = nTemp;

            pEle = child->FirstChildElement("PITCH", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.pitch = nTemp;

            pEle = child->FirstChildElement("ROLL", false);
            if (pEle == NULL)
            {
                params.clear();
                return false;
            }
            pEle->GetText(&nTemp);
            param.roll = nTemp;

            params.push_back(param);
        }
    }

    return true;
}

static bool loadPhotoParamFromXML2(const std::string& fileName, std::vector<PhotoParam>& params)
{
    params.clear();

    Document doc;
    try
    {
        doc.LoadFile(fileName);
    }
    catch (...)
    {
        return false;
    }

    Element* pRoot = doc.FirstChildElement("Root", false);
    if (pRoot == NULL)
    {
        params.clear();
        return false;
    }

    Element* pPos = pRoot->FirstChildElement("POSITION", false);
    Element* pVideo = pRoot->FirstChildElement("VIDEO", false);
    if (pPos == NULL && pVideo == NULL)
    {
        params.clear();
        return false;
    }

    for (ticpp::Iterator<ticpp::Element> child(pPos ? pPos : pVideo, pPos ? "POSITION" : "VIDEO"); child != child.end(); child++)
    {
        const Element* pEle = NULL;

        PhotoParam param;
        memset(&param, 0, sizeof(PhotoParam));

        double nTemp;

        pEle = child->FirstChildElement("IMAGETYPE", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.imageType = nTemp;

        pEle = child->FirstChildElement("CROPMODE", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropMode = nTemp;

        pEle = child->FirstChildElement("HFOV", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.hfov = nTemp;

        pEle = child->FirstChildElement("VFOV", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.vfov = nTemp;

        pEle = child->FirstChildElement("Alpha", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.alpha = nTemp;

        pEle = child->FirstChildElement("Beta", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.beta = nTemp;

        pEle = child->FirstChildElement("Gamma", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.gamma = nTemp;

        pEle = child->FirstChildElement("ShiftX", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shiftX = nTemp;

        pEle = child->FirstChildElement("ShiftY", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shiftY = nTemp;

        pEle = child->FirstChildElement("ShearX", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shearX = nTemp;

        pEle = child->FirstChildElement("ShearY", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.shearY = nTemp;

        pEle = child->FirstChildElement("CROPLEFT", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropX = nTemp;

        pEle = child->FirstChildElement("CROPRIGHT", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropWidth = nTemp - param.cropX;

        pEle = child->FirstChildElement("CROPTOP", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropY = nTemp;

        pEle = child->FirstChildElement("CROPBOTTOM", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.cropHeight = nTemp - param.cropY;

        pEle = child->FirstChildElement("CircleX", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.circleX = nTemp;

        pEle = child->FirstChildElement("CircleY", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.circleY = nTemp;

        pEle = child->FirstChildElement("CircleR", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.circleR = nTemp;

        pEle = child->FirstChildElement("YAW", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.yaw = nTemp;

        pEle = child->FirstChildElement("PITCH", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.pitch = nTemp;

        pEle = child->FirstChildElement("ROLL", false);
        if (pEle == NULL)
        {
            params.clear();
            return false;
        }
        pEle->GetText(&nTemp);
        param.roll = nTemp;

        params.push_back(param);
    }

    return true;
}

void loadPhotoParamFromXML(const std::string& fileName, std::vector<PhotoParam>& params)
{
    if (loadPhotoParamFromXML2(fileName, params))
        return;
    loadPhotoParamFromXML1(fileName, params);
}

inline bool isInfoLine(const std::string& line)
{
    return (!line.empty() && line[0] != '#');
}

static std::string getValueString(const std::string& line, std::string::size_type posBeg)
{
    std::string::size_type posEnd;
    posEnd = line.find(' ');
    if (posEnd == std::string::npos)
        return line.substr(posBeg);
    return line.substr(posBeg, posEnd - posBeg);
}

static void getValueStrings(const std::string& line, std::string::size_type posBeg, std::vector<std::string>& valStrs)
{
    valStrs.clear();
    std::string valueString = getValueString(line, posBeg);
    std::vector<int> pos;
    pos.push_back(-1);
    int len = valueString.size();
    for (int i = 0; i < len; i++)
    {
        if (valueString[i] == ',')
            pos.push_back(i);
    }
    pos.push_back(len);
    int numVals = pos.size() - 1;
    for (int i = 0; i < numVals; i++)
        valStrs.push_back(valueString.substr(pos[i] + 1, pos[i + 1] - pos[i] - 1));
}

inline double toDouble(const std::string& str)
{
    std::stringstream sstrm;
    double val;
    sstrm << str;
    sstrm >> val;
    return val;
}

inline double getValue(const std::string& line, std::string::size_type posBeg)
{
    return toDouble(getValueString(line, posBeg));
}

static void getValues(const std::string& line, std::string::size_type posBeg, std::vector<double>& vals)
{
    vals.clear();
    std::vector<std::string> valStrs;
    getValueStrings(line, posBeg, valStrs);
    int numVals = valStrs.size();
    vals.resize(numVals);
    for (int i = 0; i < numVals; i++)
        vals[i] = toDouble(valStrs[i]);
}

static bool tryParseHFov(const std::string& line, double& fov)
{
    if (line.empty() || line[0] != 'o')
        return false;

    std::string::size_type posV = line.find('v');
    if (posV == std::string::npos)
        return false;

    fov = getValue(line, posV + 1);
    return true;
}

static bool tryParseGlobalPhotoParam(const std::string& line, PhotoParam& param)
{
    if (line.empty() || line[0] != 'o')
        return false;

    std::string::size_type posV, posA, posB, posC, posD, posE, posG, posT;
    posV = line.find('v');
    posA = line.find('a');
    posB = line.find('b');
    posC = line.find('c');
    posD = line.find('d');
    posE = line.find('e');
    posG = line.find('g');
    posT = line.find('t');
    if (posV == std::string::npos ||
        posA == std::string::npos ||
        posB == std::string::npos ||
        posC == std::string::npos ||
        posD == std::string::npos ||
        posE == std::string::npos ||
        posG == std::string::npos ||
        posT == std::string::npos)
        return false;

    memset(&param, 0, sizeof(param));

    param.hfov = getValue(line, posV + 1);
    param.alpha = getValue(line, posA + 1);
    param.beta = getValue(line, posB + 1);
    param.gamma = getValue(line, posC + 1);
    param.shiftX = getValue(line, posD + 1);
    param.shiftY = getValue(line, posE + 1);
    param.shearX = getValue(line, posG + 1);
    param.shearY = getValue(line, posT + 1);

    return true;
}

static bool tryParseIndividualPhotoParam(const std::string& line, PhotoParam& param)
{
    if (line.empty() || line[0] != 'o')
        return false;

    std::string::size_type posF, posY, posP, posR, posC;
    posF = line.find('f');
    posY = line.find('y');
    posP = line.find('p');
    posR = line.find('r');
    posC = line.find('C');
    if (posF == std::string::npos ||
        posY == std::string::npos ||
        posP == std::string::npos ||
        posR == std::string::npos/* ||
        posC == std::string::npos*/)
        return false;

    memset(&param, 0, sizeof(param));

    // panotools projection format,
    // 0 - rectilinear(normal lenses)
    // 1 - Panoramic(Scanning cameras like Noblex)
    // 2 - Circular fisheye
    // 3 - full - frame fisheye
    // 4 - PSphere, equirectangular
    // program projection format
    // itNormalRectilinear = 0
    // itFisheyeFullFrame = 1
    // itFisheyeDrum = 2
    // itFisheyeCircle = 3
    double ptsImageTypeDouble = getValue(line, posF + 1);
    int ptsImageType = ptsImageTypeDouble + 0.5;
    if (ptsImageType == PTImageTypeRectlinear)
        param.imageType = PhotoParam::ImageTypeRectlinear;
    else if (ptsImageType == PTImageTypeCircularFishEye)
        param.imageType = PhotoParam::ImageTypeCircularFishEye;
    else if (ptsImageType == PTImageTypeFullFrameFishEye)
        param.imageType = PhotoParam::ImageTypeFullFrameFishEye;

    param.yaw = getValue(line, posY + 1);
    param.pitch = getValue(line, posP + 1);
    param.roll = getValue(line, posR + 1);

    if (posC != std::string::npos)
    {
        std::vector<double> vals;
        getValues(line, posC + 1, vals);
        if (vals.size() != 4)
            return false;

        param.cropX = vals[0];
        param.cropY = vals[2];
        param.cropWidth = vals[1] - vals[0];
        param.cropHeight = vals[3] - vals[2];
    }

    return true;
}

void loadPhotoParamFromPTS(const std::string& fileName, std::vector<PhotoParam>& params)
{
    params.clear();

    std::ifstream fstrm(fileName);
    if (!fstrm.is_open()) return;

    bool validFile = false;
    std::string line;
    while (true)
    {
        if (fstrm.eof())
            break;

        std::getline(fstrm, line);
        if (line.empty())
            continue;

        if (line.find("ptGui project file") != std::string::npos)
        {
            validFile = true;
            break;
        }
    }
    if (!validFile)
        return;

    PhotoParam globalParam;
    bool globalSet = false;
    double fov = -1;
    while (true)
    {
        if (fstrm.eof())
            break;

        std::getline(fstrm, line);
        if (line.empty())
            continue;

        if (isInfoLine(line) && !globalSet)
        {
            globalSet = tryParseGlobalPhotoParam(line, globalParam);
            // Add the following condition continue to prevent line
            // from being parsed twice
            if (globalSet)
                continue;
        }            

        PhotoParam localParam;
        if (isInfoLine(line) && tryParseIndividualPhotoParam(line, localParam))
            params.push_back(localParam);
    }

    //if (fov < 0)
    if (!globalSet)
    {
        params.clear();
        return;
    }

    int numParams = params.size();
    for (int i = 0; i < numParams; i++)
    {
        params[i].hfov = globalParam.hfov;
        params[i].alpha = globalParam.alpha;
        params[i].beta = globalParam.beta;
        params[i].gamma = globalParam.gamma;
        params[i].shiftX = globalParam.shiftX;
        params[i].shiftY = globalParam.shiftY;
        params[i].shearX = globalParam.shearX;
        params[i].shearY = globalParam.shearY;
    }
    //    params[i].hfov = fov;
}

bool loadPhotoParams(const std::string& cameraParamFile, std::vector<PhotoParam>& params)
{
    std::string::size_type pos = cameraParamFile.find_last_of(".");
    if (pos == std::string::npos)
    {
        printf("Error in %s, file does not have extention\n", __FUNCTION__);
        return false;
    }
    std::string ext = cameraParamFile.substr(pos + 1);
    if (ext == "pts")
        loadPhotoParamFromPTS(cameraParamFile, params);
    else
        loadPhotoParamFromXML(cameraParamFile, params);
    return params.size() > 0;
}

void exportPhotoParamToXML(const std::string& fileName, const std::vector<PhotoParam>& params)
{
    Document doc;
    Element* pElement = new Element("Root");
    doc.LinkEndChild(pElement);

    for (int i = 0; i < params.size(); ++i)
    {
        Element* pPoisitionInfo = new Element("POSITION");

        Element* pEle = NULL;
        pEle = new Element("IMAGETYPE");
        pEle->SetText(params[i].imageType);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CROPMODE");
        pEle->SetText(params[i].cropMode);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("HFOV");
        pEle->SetText(params[i].hfov);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("VFOV");
        pEle->SetText(params[i].vfov);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pEle = new Element("Alpha");
        pEle->SetText(params[i].alpha);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("Beta");
        pEle->SetText(params[i].beta);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("Gamma");
        pEle->SetText(params[i].gamma);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pEle = new Element("ShiftX");
        pEle->SetText(params[i].shiftX);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("ShiftY");
        pEle->SetText(params[i].shiftY);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("ShearX");
        pEle->SetText(params[i].shearX);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("ShearY");
        pEle->SetText(params[i].shearY);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pEle = new Element("CROPLEFT");
        pEle->SetText(params[i].cropX);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CROPRIGHT");
        int crop_right = params[i].cropX + params[i].cropWidth;
        pEle->SetText(crop_right);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CROPTOP");
        pEle->SetText(params[i].cropY);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CROPBOTTOM");
        int crop_bottom = params[i].cropY + params[i].cropHeight;
        pEle->SetText(crop_bottom);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pEle = new Element("CircleX");
        pEle->SetText(params[i].circleX);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CircleY");
        pEle->SetText(params[i].circleY);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("CircleR");
        pEle->SetText(params[i].circleR);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pEle = new Element("YAW");
        pEle->SetText(params[i].yaw);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("PITCH");
        pEle->SetText(params[i].pitch);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;
        pEle = NULL;
        pEle = new Element("ROLL");
        pEle->SetText(params[i].roll);
        pPoisitionInfo->LinkEndChild(pEle);
        if (pEle)
            delete pEle;

        pEle = NULL;
        pElement->LinkEndChild(pPoisitionInfo);
        if (pPoisitionInfo)
            delete pPoisitionInfo;
        pPoisitionInfo = NULL;
    }

    doc.SaveFile(fileName);

    return;
}

const static double degOverRad = 180.0 / 3.1415926535898;
const static double radOverDeg = 3.1415926535898 / 180.0;

void rotateCamera(PhotoParam& param, double yaw, double pitch, double roll)
{
    /*
    Mtx33 rot;
    rot.SetRotationPT(yaw, pitch, roll);
    double currYaw = param.yaw * radOverDeg;
    double currPitch = param.pitch * radOverDeg;
    double currRoll = param.roll * radOverDeg;
    Mtx33 pos;
    pos.SetRotationPT(currYaw, currPitch, currRoll);
    pos = rot * pos;
    pos.GetRotationPT(currYaw, currPitch, currRoll);
    param.yaw = currYaw * degOverRad;
    param.pitch = currPitch * degOverRad;
    param.roll = currRoll * degOverRad;
    */

    cv::Matx33d rot;
    setRotationRM(rot, yaw, pitch, roll);
    double currYaw = param.yaw * radOverDeg;
    double currPitch = param.pitch * radOverDeg;
    double currRoll = param.roll * radOverDeg;
    cv::Matx33d pos;
    setRotationRM(pos, currYaw, currPitch, currRoll);
    pos = rot * pos;
    getRotationRM(pos, currYaw, currPitch, currRoll);
    param.yaw = currYaw * degOverRad;
    param.pitch = currPitch * degOverRad;
    param.roll = currRoll * degOverRad;
}

void rotateCameras(std::vector<PhotoParam>& params, double yaw, double pitch, double roll)
{
    int num = params.size();
    for (int i = 0; i < num; i++)
        rotateCamera(params[i], yaw, pitch, roll);
}

void rotatePhotoParamInXML(const std::string& src, const std::string& dst, double yaw, double pitch, double roll)
{
    std::vector<PhotoParam> params;
    loadPhotoParamFromXML(src, params);
    rotateCameras(params, yaw, pitch, roll);
    exportPhotoParamToXML(dst, params);
}
