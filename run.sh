#!/bin/bash
n=0
while [ $n -lt 10 ]
do
    ./numa.bin >> oblivious.log
    let n=n+1
done
