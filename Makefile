CC=g++
INC=-I/usr/include/eigen3
CFLAGS=-c -Wall -std=c++11 $(INC)
LDFLAGS=
LDLIBS=
SOURCEPATH=./src
SOURCES=$(SOURCEPATH)/main.cpp \
	$(SOURCEPATH)/Layer.cpp


OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=digits_recognizer

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	    $(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

.cpp.o:
	    $(CC) $(CFLAGS) $< -o $@

clean:
			rm $(SOURCEPATH)/*.o
			rm $(EXECUTABLE)
