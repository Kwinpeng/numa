CC = g++
FLAGS = -fopenmp

INCS = -I.
LIBS = -lnuma

numa.bin: numa.cpp
	$(CC) $(FLAGS) $< -o $@ $(LIBS)

clean:
	@rm -f numa.bin
