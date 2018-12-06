#include "../include/importheaders.h"
#include "../include/action.h"
#include "../include/server.h"
using namespace std;

int main(int argc, char * argv[])
{

	NetworkServer::Server s(100);
	s.Start( string(argv[1]) );

	return 0;
}
