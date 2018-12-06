#ifndef NETWORKSERVERRESPONSEHANDLER_H
#define NETWORKSERVERRESPONSEHANDLER_H

#include "./importheaders.h"
#include "../include/synchronizationdata.h"
#include "../include/action.h"
using namespace std;

namespace NetworkServer {


class ResponseHandler
{
	private:
		static void RemoveConnectionDetails
				(SynchronizationData *, int index);

		static void DisplayDiagnosticMessages(string nm, Action *);

	public:

		ResponseHandler();
		~ResponseHandler();

		static void * Start(void * inp);



};

}

#endif
