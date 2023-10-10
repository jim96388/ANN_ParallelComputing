all: autoEncoder.exe autoEncoderOMP.exe autoEncoderOCL.exe 

CC=g++
CFLAGS=-Wall -O3 -ffast-math

OPENCV=-DOPENCV -I/usr/include/opencv4 
lOPENCV=-lopencv_core -lopencv_imgcodecs

%OMP.o: src/%.cpp
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

%OCL.o: src/%.cpp
	$(CC) $(CFLAGS) -DOPENCL -c $< -o $@

%.o: src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: src/main.cpp
	$(CC) $(CFLAGS) $(OPENCV) -c $< -o $@ 

ANNOCL.o: src/ANNOCL.cpp
	$(CC) $(CFLAGS) -DOPENCL -c $< -o $@

ANNOMP.o: src/ANNOMP.cpp
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

# $^ is a special macro containing all dependencies
autoEncoder.exe: main.o ANN.o CIFAR10.o MNIST.o
	$(CC) $(CFLAGS) -o $@ $^  $(lOPENCV)

autoEncoderOMP.exe: main.o ANNOMP.o CIFAR10.o MNIST.o
	$(CC) $(CFLAGS) -fopenmp -o $@ $^  $(lOPENCV)

autoEncoderOCL.exe: mainOCL.o ANNOCL.o CIFAR10OCL.o MNISTOCL.o GPU.o
	$(CC) $(CFLAGS) -o $@ $^ -lOpenCL


M: autoEncoder.exe
	./$< 0 2 10 1.0 
# 模型結構 批次數量 訓練次數 學習率 

C: autoEncoder.exe
	./$< 0 1 1 1.0 1 >> log
# 模型結構 批次數量 訓練次數 學習率 讀取CIFAR10

check: autoEncoder.exe
	./$< 

check100: autoEncoder.exe
	for i in {1..100}; do \
	 		./$< ;\
	done

checkomp: autoEncoderOMP.exe
	./$< 

checkompT: autoEncoderOMP.exe
	./$< 1 4 1 0.4

checkocl: autoEncoderOCL.exe
	srun -w gpu0 ./$< 

clean:
	rm *o *exe *tiff log


	# for i in {1..100};\
	# 	do\
	# 		srun -w gpu0 ./$< 
	# 	done
	# for i in {1..100}; 
	# 	do \
	# 		./$< ;\
	# done