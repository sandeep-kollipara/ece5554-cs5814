#!/bin/bash

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <train.part) train.part | tr -d '\t' > train.txt
paste <(awk "{print \"$PWD\"}" <val.part) val.part | tr -d '\t' > val.txt
