CC = icpc
FLAGS = -fopenmp \
	   	-mavx \
		-msse2 \
		#-O2 -ftree-vectorize -msse2 -ftree-vectorizer-verbose=5

INCS = -I.

LIBS = -lnuma

numa.bin: numa.cpp
	$(CC) $(FLAGS) $< -o $@ $(LIBS)

clean:
	@rm -f numa.bin
