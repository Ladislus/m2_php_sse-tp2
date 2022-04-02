# Project SSE

## Members
 - WALCAK Ladislas
 - QUETIER Thomas

## Compilation Options
The CMake project contains multiple options to enable or disable features.
 - **VERBOSE**: Enables verbose output (Defaulted to ON in CMake debug mode, OFF else).
 - **CHECK**: Enables the verification of the result matrix using naive function (Defaulted to ON in CMake debug mode, OFF else).
 - **SSE**: Enable the use of SSE functions instead of naive function (Defaulted to ON).  
   To use naive approach, set it to OFF along with BONUS.
 - **BONUS**: Enable the use of SSE block multiplication function instead of naive function (Defaulted to OFF).  
   If both SSE and BONUS are enabled, SSE has the priority.  
   To use naive approach, set it to OFF along with SSE.

## Change Input
You can change the matrix size by changing the value of the `dim` variable in the [main](main.cpp) file.  
Matrix is a square matrix of size `dim` x `dim`, and `dim` is multiplied by 4 (`ELEM_SIZE` macro), in order to work with float registers.  