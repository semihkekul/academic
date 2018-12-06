#include "../include/connectionmonitor.h"

namespace NetworkServer {

ConnectionMonitor::ConnectionMonitor()
{
}

ConnectionMonitor::~ConnectionMonitor()
{
}

void * ConnectionMonitor::Start(void * inp)
{
	SynchronizationData * syncData = (SynchronizationData*) inp;

	struct timeval currentTime;

	while(1)
	{
		gettimeofday( &currentTime, NULL);

		for( unsigned int i = 0; i < syncData->lastResponseTime.size() ; i++)
		{
			// check for time out
			if( (currentTime.tv_sec - syncData->lastResponseTime.at(i).tv_sec) > TIME_OUT_LIMIT )
			{
				syncData->timeOutFlags.at(i) = true;

				SendMessage("SERVER >>> Your connection has timed out due to inactivity. You can spectate the game but can't perform any action\n",
								   				syncData->connectedSocks.at(i) );
			}
			else if( (currentTime.tv_sec - syncData->lastResponseTime.at(i).tv_sec) > (4*TIME_OUT_LIMIT/5) )
			{

				SendMessage("SERVER >>> Your connection is about to time out soon. You have to perform an action immediately.\n",
								   syncData->connectedSocks.at(i) );
			}
		}

		// check every 20 seconds
		sleep(10);

	}
}

void ConnectionMonitor::SendMessage(string message, int sock)
{

	Action a;
	a.type = Message;
	a.theAction.message.msg = strdup( message.c_str() );
	a.theAction.message.msgLength = strlen( a.theAction.message.msg );

	a.WriteToSocket_Linux( sock );

	a.Deallocate();
}

}
