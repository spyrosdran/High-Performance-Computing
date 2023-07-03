## How to compile and run the code
✨ First, you should install the mpich package:
```apt install mpich```

👉 Then, to compile a file:
```mpicc <file> -o <output_file_name>```

👉 To run the code:
```mpiexec -n <number_of_processes> <output_file_name> <...>```
