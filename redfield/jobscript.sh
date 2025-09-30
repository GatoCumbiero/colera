#! /bin/bash
file='redfield.py' # 'dynamics_hamiltonian1.py' # adjust to run other script
# activate python environment, make a wrapper for it (otherwise command line arguments are passed to it)
wrappersource() {
    source /usr/local/python3/bin/activate
}
wrappersource
# run script with command line arguments (relevant for job arrays) $1=Cluster, $2=Process 
python3 $file $1
exit