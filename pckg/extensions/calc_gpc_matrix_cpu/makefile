CC=gcc

all: libgpc.a

libgpc.a: gpc.o
	ar rcs $@ $<
	rm $^

gpc.o: gpc.c
	$(CC) -c -Wall -fpic -fopenmp $^

clean:
	rm -f libgpc.a
