numa.bin: numa.cpp
	g++ -fopenmp numa.cpp -o numa.bin -lnuma

clean:
	@rm -f numa.bin
