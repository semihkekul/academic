#ifndef NETWORKSERVERSERVER_H
#define NETWORKSERVERSERVER_H

#include "./importheaders.h"
#include "../include/synchronizationdata.h"
#include "../include/connectionlistener.h"
#include "../include/connectionmonitor.h"
using namespace std;

namespace NetworkServer {


class Server
{
    private:

	SynchronizationData * syncData;

	void OpenConnection();
	void CloseConnection();


    public:

	Server( int maxCapacity = 10 );
	~Server();

	// start executing the server
	void Start(string connectionPort);

};

}

#endif
