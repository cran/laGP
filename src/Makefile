all: laGP$(SHLIB_EXT)

allgpu: alc_gpu.o laGP$(SHLIB_EXT)

laGP.so:
	${R_HOME}/bin/R CMD SHLIB -o laGP$(SHLIB_EXT) *.c

alc_gpu.o:
	nvcc -arch=sm_20 -c -Xcompiler -fPIC alc_gpu.cu -o alc_gpu.o

clean:
	rm -rf *.o *.so
