#ifndef NETWORKSERVERCONNECTIONMONITOR_H
#define NETWORKSERVERCONNECTIONMONITOR_H

#include "./importheaders.h"
#include "../include/synchronizationdata.h"
#include "../include/action.h"
using namespace std;


namespace NetworkServer {

class ConnectionMonitor{

public:
    ConnectionMonitor();
    ~ConnectionMonitor();

	static void * Start(void * inp);
	static void SendMessage(string message, int sock);

};

}

#endif
