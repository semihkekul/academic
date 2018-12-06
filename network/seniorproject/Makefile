seniorproject: server.o connectionlistener.o responsehandler.o synchronizationdata.o seniorproject.o action.o connectionmonitor.o
	g++ -o seniorproject *.o -Wall -lpthread -g

server.o :
	g++ -c ./src/server.cpp -Wall -g

connectionlistener.o :
	g++ -c ./src/connectionlistener.cpp -Wall -g

connectionmonitor.o :
		g++ -c ./src/connectionmonitor.cpp -Wall -g

responsehandler.o :
	g++ -c ./src/responsehandler.cpp -Wall -g

synchronizationdata.o :
	g++ -c ./src/synchronizationdata.cpp -Wall -g

action.o :
	g++ -c ./src/action.cpp -Wall -g

seniorproject.o :
	g++ -c ./src/seniorproject.cpp -Wall -g




clean:
	rm *.o seniorproject
