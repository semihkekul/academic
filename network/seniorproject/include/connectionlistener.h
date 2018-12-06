#ifndef NETWORKSERVERCONNECTIONLISTENER_H
#define NETWORKSERVERCONNECTIONLISTENER_H

#include "./importheaders.h"
#include "../include/synchronizationdata.h"
#include "../include/responsehandler.h"
using namespace std;

namespace NetworkServer {


class ConnectionListener
{
	public:

	ConnectionListener();
	~ConnectionListener();

	static void * Start(void * inp);

	static bool Authenticate(int connectedSock, SynchronizationData * sd );

};

}

#endif
