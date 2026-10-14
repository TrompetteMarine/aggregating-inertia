# Compile the C kernels of the reference pipeline next to the Python solvers.
CC ?= gcc
CFLAGS ?= -O2 -shared -fPIC

all: code/solvers/march.so code/solvers/libsweep.so code/solvers/libmc.so

code/solvers/march.so: code/solvers/march.c
	$(CC) $(CFLAGS) -o $@ $<
code/solvers/libsweep.so: code/solvers/kernel.c
	$(CC) $(CFLAGS) -o $@ $<
code/solvers/libmc.so: code/solvers/mc.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f code/solvers/*.so
