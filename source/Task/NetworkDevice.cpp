#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include "IPCamSDKDLL.h"

#include <vector>
#include <string>

static void getLocalHostIPs(std::vector<std::string>& IPs)
{
    IPs.clear();

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
        return;

    struct hostent *remoteHost;
    struct in_addr addr;

    remoteHost = gethostbyname(NULL);

    if (remoteHost == NULL)
        return;

    if (remoteHost->h_addrtype == AF_INET)
    {
        int i = 0;
        while (remoteHost->h_addr_list[i] != 0)
        {
            addr.s_addr = *(u_long *)remoteHost->h_addr_list[i++];
            IPs.push_back(inet_ntoa(addr));
        }
    }
}

void __stdcall searchCallback(DEVICE_ALL_INFO *pDevAllInfo, void *pUserData)
{
    //printf("ip = %s\n", pDevAllInfo->szIP);
    std::vector<std::string>& IPs = *(std::vector<std::string>*)pUserData;
    bool insert = true;
    for (int i = 0; i < IPs.size(); i++)
    {
        if (IPs[i] == pDevAllInfo->szIP)
        {
            insert = false;
            break;
        }
    }
    if (insert)
        IPs.push_back(pDevAllInfo->szIP);
}

void listNetworkDevices(std::vector<std::string>& urls)
{
    urls.clear();

    std::vector<std::string> hostIPs;
    getLocalHostIPs(hostIPs);
    if (hostIPs.empty())
        return;

    int ret = 0;
    ret = SDK_Init();
    if (ret != 0)
        return;
    int numHostIPs = hostIPs.size();
    for (int i = 0; i < numHostIPs; i++)
        ret = SDK_SearchDevices(searchCallback, &urls, (char *)hostIPs[i].c_str(), 1);

    SDK_Cleanup();
}