#include "../include/synchronizationdata.h"

namespace NetworkServer {


SynchronizationData::SynchronizationData( int maxCapacity )
{
	sock = 0;
	maxNoClients = maxCapacity;
	noClients = 0;

	pthread_mutex_init( &noClientsMutex, NULL);

	pthread_t thr;
	for(int i = 0; i < maxCapacity; i++)
		responseHandlerThreads.push_back(thr);

	sem_init( &controlFlowSemaphore, 0, 0 );

}


SynchronizationData::~SynchronizationData()
{
	pthread_mutex_destroy(&noClientsMutex);

	sem_destroy( &controlFlowSemaphore );
}


int SynchronizationData::getNoClients(int offset)
{
	int val;

	pthread_mutex_lock(&noClientsMutex);

	val = noClients + offset;

	pthread_mutex_unlock(&noClientsMutex);

	return val;
}

void SynchronizationData::updateNoClients( int offset )
{
	pthread_mutex_lock(&noClientsMutex);

	noClients += offset;

	pthread_mutex_unlock(&noClientsMutex);
}

void SynchronizationData::seizeControl()
{
	sem_wait(&controlFlowSemaphore);
}

void SynchronizationData::releaseControl()
{
	sem_post(&controlFlowSemaphore);
}

unsigned int SynchronizationData::getIndex( int cs )
{
	for (unsigned int i = 0; i < connectedSocks.size() ; i++)
	{
		if( connectedSocks.at(i) == cs )
			return i;
	}

	return 0;
}

}
