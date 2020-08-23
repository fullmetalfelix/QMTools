
echo compiling CUDA convolver...

echo arg1 is $1
arch=$1
if [ -z $arch ]
then
	arch="sm_60"
fi


nvcc -O3 --resource-usage -o convolve.exe cube.cu kernel.cu vectors.cu convolve.cu main.cu  -rdc=true -arch=$arch -lm -DDEBUGPRINT
#nvcc -O3 -c convolve.cu --resource-usage # -arch=sm_60


echo CUDA convolver compiled!

#echo compiling address tester...
#gcc -O3 -c setup.c
#gcc -O3 -c addresser.c
#gcc -O3 -o addresser.exe setup.o addresser.o 
