#!/usr/bin/env python

python -u entry.py -task char2ir -bl ebu3b -nl 10 -t ebu3b -post 0 &
python -u entry.py -task char2ir -bl ebu3b -nl 10 -t ebu3b -post 1 &
python -u entry.py -task char2ir -bl ebu3b -nl 10 -t ebu3b -post 2 &
python -u entry.py -task char2ir -bl ebu3b -nl 10 -t ebu3b -post 3 &
python -u entry.py -task char2ir -bl ap_m,ebu3b -nl 200,10 -t ebu3b -post 0 &
python -u entry.py -task char2ir -bl ap_m,ebu3b -nl 200,10 -t ebu3b -post 1 &
python -u entry.py -task char2ir -bl ap_m,ebu3b -nl 200,10 -t ebu3b -post 2 &
python -u entry.py -task char2ir -bl ap_m,ebu3b -nl 200,10 -t ebu3b -post 3
