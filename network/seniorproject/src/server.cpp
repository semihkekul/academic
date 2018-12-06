#include "../include/server.h"

namespace NetworkServer {

Server::Server(int maxCapacity)
{
	syncData = new SynchronizationData(maxCapacity);
}


Server::~Server()
{
	delete syncData;
}

void Server::OpenConnection()
{
	/* create the socket name */
	syncData->sock = socket( AF_INET, SOCK_STREAM, 0 );
	if( syncData->sock < 0 ){

		perror("error : socket() .\n");
		CloseConnection( );
		exit(EXIT_FAILURE);
	}

	/* initialize the socket address */
	bzero( &syncData->serv_name, sizeof(syncData->serv_name) );
	syncData->serv_name.sin_family = AF_INET;
	syncData->serv_name.sin_port = htons( atoi(syncData->port) );

	/* create a socket by binding the name to the address */
	if( bind( syncData->sock, (struct sockaddr *)&syncData->serv_name,
	    	sizeof(syncData->serv_name) ) < 0 )
	{
		perror("Naming channel.\n");
		CloseConnection( );
		exit(EXIT_FAILURE);
	}


}

void Server::CloseConnection()
{
	close( syncData->sock );
}

void Server::Start( string connectionPort )
{
	syncData->port[0] = connectionPort[0];
	syncData->port[1] = connectionPort[1];
	syncData->port[2] = connectionPort[2];
	syncData->port[3] = connectionPort[3];

	// open Connection
	OpenConnection();

	// create connection listener
	ConnectionListener conList;

	syncData->idConnectionListenerD =
	pthread_create( &syncData->connectionListenerThread, NULL,
			conList.Start, (void*)syncData );

	// create connection monitor
	ConnectionMonitor conMon;

	pthread_create( &syncData->connectionMonitorThread, NULL,
					 conMon.Start, (void*)syncData );


	// wait for the connection handler to exit
	pthread_join( syncData->connectionListenerThread, NULL);
	// wait for the connection monitor to exit
	pthread_join( syncData->connectionMonitorThread, NULL);
}



}
