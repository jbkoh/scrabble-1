#!/usr/bin/env python

(nohup python -u entry.py -task char2ir -bl ap_m -nl 10 -t ap_m -post lstm0 > nohup.lstm0.out) &
(nohup python -u entry.py -task char2ir -bl ap_m -nl 10 -t ap_m -post lstm1 > nohup.lstm1.out) &
(nohup python -u entry.py -task char2ir -bl ap_m -nl 10 -t ap_m -post lstm2 > nohup.lstm2.out) &
(nohup python -u entry.py -task char2ir -bl ap_m -nl 10 -t ap_m -post lstm3 > nohup.lstm3.out) &
