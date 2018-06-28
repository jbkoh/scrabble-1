#!/usr/bin/env python

python -u entry.py -task scrabble -bl ebu3b -nl 10 -t ebu3b -ub true -ut true -neg true -ct MLP #&
python -u entry.py -task scrabble -bl ebu3b -nl 10 -t ebu3b -ub true -ut true -neg true -ct MLP;
python -u entry.py -task scrabble -bl ap_m,ebu3b -nl 200,10 -t ebu3b -ub true -ut true -neg true -ct MLP;
python -u entry.py -task scrabble -bl ebu3b,sdh -nl 200,10 -t sdh -ub true -ut true -neg true -ct MLP #&
python -u entry.py -task scrabble -bl sdh,ebu3b -nl 200,10 -t ebu3b -ub true -ut true -neg true -ct MLP;
