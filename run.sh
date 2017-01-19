#!/bin/bash 

dir_name=$1                 # directory name is passed as first argument
program_file=bin/$1         # program executable file name equals to dir name
result_file="out.txt"       # file where results are taken from

cd $dir_name

# compile if executable is older than source or doesn't exist
if [[ main.c -nt $program_file ]]
    then
        rm $program_file
        mkdir bin
        mpicc -W -Wall -std=c99 main.c -o $program_file # <-- compilation
fi


# clear previous run results
rm -f $result_file

mpiexec -np 4 ./$program_file   # <-- execute program (test run)

# print the results from file
printf "\nResults from $result_file:\n"
cat $result_file
printf "\n"
