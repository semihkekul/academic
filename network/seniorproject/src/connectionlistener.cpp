#include "../include/connectionlistener.h"

namespace NetworkServer {

ConnectionListener::ConnectionListener()
{

}


ConnectionListener::~ConnectionListener()
{

}

void * ConnectionListener::Start(void * inp)
{
	SynchronizationData * syncData = (SynchronizationData*) inp;

	// set the connection ready for listening up to
	// maxNoClients pending connection requests
	listen(syncData->sock, syncData->maxNoClients);

	socklen_t len = sizeof( syncData->serv_name );

	// listen up to the limit
	while( syncData->getNoClients(0) < syncData->maxNoClients )
	{
		int connectedSock =
			accept( syncData->sock,
				(struct sockaddr *) &syncData->serv_name, &len );

		if( connectedSock != -1 )
		{


			// update the connected sock
			syncData->connectedSocks.push_back( connectedSock );

			// set response time for this player
			struct timeval tv;
			gettimeofday( &tv, NULL);
			syncData->lastResponseTime.push_back( tv );
			// set its flag
			syncData->timeOutFlags.push_back(false);

			// check if this player can enter
			Authenticate(connectedSock, syncData);


			// debug
			cout << syncData->clientNames.at(
					syncData->getIndex( connectedSock ) )
					<< " connected.\n";
			cout.flush();

			// create a handler for the created connection

			ResponseHandler respHand;
			ResponseHandlerInput inp;
			inp.sd = syncData;
			inp.cs = connectedSock;

			syncData->responseHandlerThreadsIds.push_back
				(
				 pthread_create(
					&syncData->responseHandlerThreads.at( syncData->getNoClients(0) ),
					 NULL, respHand.Start, (void*) &inp )
				);

			syncData->updateNoClients(1);
		}

	}

	return inp;
}


bool ConnectionListener::Authenticate( int connectedSock, SynchronizationData * syncData )
{
	// is this player registered
	//bool registered = false;

	// read name
	char nm[50];
	read(connectedSock, nm, 50*sizeof(char));

	// validate it by querying the database
	syncData->clientNames.push_back(string(nm));


	return true;
}


}
