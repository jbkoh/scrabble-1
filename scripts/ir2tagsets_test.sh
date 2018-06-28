#!/bin/bash

# sample aug    no source   MLP
#python -u entry.py -task ir2tagsets -bl ap_m -nl 10 -t ap_m -neg true -ub true -ut true -ct MLP -post 0
#python -u entry.py -task ir2tagsets -bl ap_m -nl 10 -t ap_m -neg false -ub false -ut true -ct MLP -post 0
#python -u entry.py -task ir2tagsets -bl ebu3b,ap_m -nl 200,10 -t ap_m -neg true -ub true -ut true -ct MLP -post 0

#python -u entry.py -task ir2tagsets -bl ap_m -nl 10 -t ap_m -neg false -ub false -ut true -ct MLP -post 1
#python -u entry.py -task ir2tagsets -bl ap_m -nl 10 -t ap_m -neg true -ub true -ut true -ct MLP -post 1
#python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 1


#python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 0 &
#python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg false -ub false -ut true -ct MLP -post 0 &
#python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 0 &

#python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg false -ub false -ut true -ct MLP -post 1 &
#python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 1 &
#python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 1 &

#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg true -ub true -ut true -ct MLP -post 0 &
#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg false -ub false -ut true -ct MLP -post 0 &
#python -u entry.py -task ir2tagsets -bl ap_m,ghc -nl 200,10 -t ghc -neg true -ub true -ut true -ct MLP -post 0 &
#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg true -ub true -ut true -ct MLP -post 2
#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg true -ub true -ut true -ct MLP -post 3 &

#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg false -ub false -ut true -ct MLP -post 1 &
#python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg true -ub true -ut true -ct MLP -post 1 &
#python -u entry.py -task ir2tagsets -bl ap_m,ghc -nl 200,10 -t ghc -neg true -ub true -ut true -ct MLP -post 1 &




python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 7 &
python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ct MLP -post 8 &
python -u entry.py -task ir2tagsets -bl ap_m,ghc -nl 200,10 -t ghc -neg true -ub true -ut true -ct MLP -post 7 &
python -u entry.py -task ir2tagsets -bl ap_m,ghc -nl 200,10 -t ghc -neg true -ub true -ut true -ct MLP -post 8 &
