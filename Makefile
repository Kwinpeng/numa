CC = icpc

FLAGS = -fopenmp \
	   	-mavx \
		-msse2 \
		-g \
		-O0 \

INCS = -I.

LIBS = -lnuma

numa.bin: numa.cpp
	$(CC) $(FLAGS) $< -o $@ $(LIBS)

clean:
	@rm -f numa.bin
