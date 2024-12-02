PROG = histogram_OpenMP

CFLAGS = -Wall -g -fopenmp  -O3 
LDLIBS = -lm

.phony: all clean

all: $(PROG)

clean:
	rm -fv $(PROG)