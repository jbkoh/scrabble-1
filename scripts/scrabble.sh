#!/usr/bin/env python


#(rm nohup.1.out; nohup python -u char2tagset_test.py -bl ebu3b,ap_m -nl 200,10 -t ap_m -ub true -ut true -entqs phrase_util > nohup.1.out; slack_notify --msg '1 done' ) &
#(rm nohup.2.out; nohup python -u char2tagset_test.py -bl ebu3b,ap_m -nl 200,10 -t ap_m -ub true -ut false -entqs phrase_util > nohup.2.out; slack_notify --msg '2 done' ) &
#(rm nohup.3.out; nohup python -u char2tagset_test.py -bl ap_m -nl 10 -t ap_m -ub true -ut true -entqs phrase_util -ct StructuredCC > nohup.3.out; slack_notify --msg '3 done' ) &
#(rm nohup.5.out; nohup python -u char2tagset_test.py -bl ap_m -nl 10 -t ap_m -ub true -ut true -entqs phrase_util -ct MLP > nohup.5.out; slack_notify --msg '5 done' ) &
#(rm nohup.4.out; nohup python -u char2tagset_test.py -bl ebu3b,ap_m -nl 200,10 -t ap_m -ub false -ut false -entqs phrase_util > nohup.4.out; slack_notify --msg '4 done' ) &

(rm nohup.6.out; nohup python -u char2tagset_test.py -bl ebu3b,ap_m -nl 200,10 -t ap_m -ub true -ut true -entqs phrase_util -ct MLP > nohup.6.out; slack_notify --msg '6 done' ) &
(rm nohup.7.out; nohup python -u char2tagset_test.py -bl ebu3b,ap_m -nl 200,10 -t ap_m -ub true -ut true -entqs phrase_util -ct StructuredCC > nohup.7.out; slack_notify --msg '7 done' ) &
