CUDAPATH = /usr/local/cuda
MQXPATH := `pwd`/../../../src
TARGET = md5

NVCC := $(CUDAPATH)/bin/nvcc
NVCCFLAGS := --compiler-options -Wall -arch=sm_20 -I$(CUDAPATH)/include \
              -I$(MQXPATH) -L$(CUDAPATH)/lib64 -L$(MQXPATH) \
              -Xlinker -rpath=$(MQXPATH)

.DEFAULT_GOAL := all
all: mqx

mqx:
	$(NVCC) -o $(TARGET) $(NVCCFLAGS) -lmqx -DUSE_MQX main.cu

nomqx:
	$(NVCC) -o $(TARGET) $(NVCCFLAGS) main.cu

clean:
	rm -f *.o $(TARGET)

test: test_mqx

test_mqx:
	@LD_PRELOAD=$(MQXPATH)/libmqx.so ./$(TARGET)

test_nomqx:
	@./$(TARGET)
