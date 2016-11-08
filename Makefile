########################################
########################################
##
## Makefile
## LINUX compilation 
##
##############################################


#FLAGS
C++FLAG = -g -std=c++11

MATH_LIBS = -lm

EXEC_DIR=.


.cc.o:
	g++ $(C++FLAG) $(INCLUDES)  -c $< -o $@

#Including
INCLUDES=  -I. 

#-->All libraries (without LEDA)
LIBS_ALL =  -L/usr/lib -L/usr/local/lib 


#First Program (ListTest)

Cpp_OBJ0=neural_net.o mnist_net.o
PROGRAM_0=mnist_net
$(PROGRAM_0): $(Cpp_OBJ0)
	g++ $(C++FLAG) -o $(EXEC_DIR)/$@ $(Cpp_OBJ0) $(INCLUDES) $(LIBS_ALL)

all: 
	make $(PROGRAM_0) 

clean:
	(rm -f *.o; rm -f mnist_net;)

(:
