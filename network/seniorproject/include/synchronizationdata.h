#ifndef NETWORKSERVERSYNCHRONIZATIONDATA_H
#define NETWORKSERVERSYNCHRONIZATIONDATA_H

#include "./importheaders.h"
using namespace std;

namespace NetworkServer {



	class SynchronizationData
	{
	    private:
		// no of clients and its mutex
		int noClients;
		pthread_mutex_t noClientsMutex;

		sem_t controlFlowSemaphore;

	    public:

		// socket and connected sockets
		int sock;
		vector<int> connectedSocks;

		// connection port
		char port[4];

		// server socket address
		struct sockaddr_in serv_name;

		// server capacity
		int maxNoClients;


		// id of the ConnectionListenerD thread
		int idConnectionListenerD;
		// thread structure for ConnectionListenerD thread
		pthread_t connectionListenerThread;

		// connection monitor thread
		pthread_t connectionMonitorThread;
		// flags to show whether a connection needs to be closed
		vector<bool>	  timeOutFlags;

		// id of each ResponseHandler thread
		vector<int> responseHandlerThreadsIds;
		// thread structure for each ResponseHandler thread
		vector<pthread_t> responseHandlerThreads;

		// last response time
		vector<timeval> lastResponseTime;

		// login names of each client
		vector<string> clientNames;


		SynchronizationData(int maxCapacity);
		~SynchronizationData();

		// get & set the no clients
		int  getNoClients(int offset );
		void updateNoClients(int offset);

		// get id index of a connection given a socket
		unsigned int getIndex(int cs);


		// assure one thread will be run to completion

		void seizeControl();
		void releaseControl();



	};


	// used to initialize input for a response Handler
	struct ResponseHandlerInput
	{
		SynchronizationData * sd;	// sync data
		int cs;				// connected sock
	};


}



#endif
