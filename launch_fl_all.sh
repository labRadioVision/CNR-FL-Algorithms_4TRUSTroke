#!/usr/bin/bash

##################################FUNCTIONS####################################
stop_mosquitto() {
    # Check if Mosquitto is running
    if [ -n "$MOSQUITTO_PID" ]; then
        # Stop Mosquitto
        echo "Stopping Mosquitto (PID: $MOSQUITTO_PID)..."
        kill -15 $MOSQUITTO_PID
        sleep 2 # Give it some time to gracefully shut down
        echo "Mosquitto stopped."
    fi
    fuser -k 1885/tcp
}

close_other_terminals() {
    # Get the ID of the current terminal
    current_tty=$(tty)
    # Get the IDs of the other terminals
    tty_list=($(ps -e -o tty= | grep -oE 'pts/[0-9]+' | sort -u))

    # Close all the other terminals
    for tty in "${tty_list[@]}"; do
        if [ /dev/$tty !=  $current_tty ]; then
            kill -9 $(ps -t $tty -o pid=)
        fi
        # TODO: Ignore broker terminal
    done
}

run_in_terminal() {
    if [ "$2" = "wait" ]; then
        gnome-terminal --wait -- bash -c "$1; sleep 7;"
    else
        gnome-terminal -- bash -c "$1; sleep 7;" 
    fi
}
##################################PARAMETERS####################################
algorithms=("FedAvg" "FedAdp" "FedProx" "Scaffold" "FedDyn" "FedDkw" "FedNova" "FedXGBllr")
runs=1

data="stroke"
samples=1000
n_iid="iid"
#################################MAIN CODE######################################
for run in $(seq 0 $((runs - 1))); do
    # Prepare tehhe data
    python -m classes.Datasets.data_generator -samples $samples -data $data -niid $n_iid -alpha 0.1
    if [ $? -eq 0 ]; then
        sleep 1

        for alg in "${algorithms[@]}"; do
            echo "Running algorithm: $alg, run: $run"
            # Call the functions
            stop_mosquitto
            close_other_terminals

            # Run the broker
            run_in_terminal "mosquitto -v -c /etc/mosquitto/conf.d/confmos.conf" 
            sleep 1 # Give some time for the broker to start           

            # Get number of clients from fl_param.py
            NUM_CLIENTS=$(grep 'NUM_CLIENTS =' classes/params/fl_param.py | awk '{print $3}')
            # Loop to run client.py in separate terminals
            for i in $(seq 0 $((NUM_CLIENTS - 1))); do
                run_in_terminal "python -m clients.client -ID $i -alg $alg -run $run" 
            done

            # Run server.py in a new terminal and wait for it to finish
            run_in_terminal "python -m servers.server -alg $alg -run $run" "wait" 
        done
    else
        # Python command failed, handle error
        echo "Error: Data partitioning failed."
    fi
done
# close all terminals
stop_mosquitto
close_other_terminals