#!/usr/bin/bash

# Function to run commands in a new terminal
run_in_terminal() {
    gnome-terminal -- bash -c "$1; exec bash"
}

# Check if Mosquitto is running
if [ -n "$MOSQUITTO_PID" ]; then
    # Stop Mosquitto
    echo "Stopping Mosquitto (PID: $MOSQUITTO_PID)..."
    kill -15 $MOSQUITTO_PID
    sleep 2 # Give it some time to gracefully shut down
    echo "Mosquitto stopped."
fi
fuser -k 1885/tcp

# Get the ID of the current terminal
current_tty=$(tty)

# Get the IDs of the other terminsls
tty_list=($(ps -e -o tty= | grep -oE 'pts/[0-9]+' | sort -u))

# Close all the other terminals
for tty in "${tty_list[@]}"; do
    if [ /dev/$tty !=  $current_tty ]; then
        kill -9 $(ps -t $tty -o pid=)
    fi
    # TODO: Ignore brocker terminal
done

#python -m classes.Datasets.data_generator -samples 500 -data stroke -niid iid -alpha 0.1
# Prepare tehhe data
#python -m classes.Datasets.data_generator -samples 1000 -data mnist -niid iid -alpha 0.1

if [ $? -eq 0 ]; then

    sleep 1

    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/home/stefano/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home/stefano/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home/stefano/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home/stefano/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    cd /home/stefano/Usevalad_projects/CNR-FL-Platform-Dockerized

    # Activate conda environment in a new terminal
    conda activate tf_env

    # Run the broker
    run_in_terminal "mosquitto -v -c /etc/mosquitto/conf.d/confmos.conf"
    # Wait for a while before launching PS
    sleep 1

    algorithm="FedXGBllr"
    run=0

    # Run PS_server_synch_v2_fordocker.py in a new terminal
    run_in_terminal "python -m servers.server -alg $algorithm -run $run"

    # Get number of clients from fl_param.py
    NUM_CLIENTS=$(grep 'NUM_CLIENTS =' classes/params/fl_param.py | awk '{print $3}')
    # Loop to run learner_fordocker.py in separate terminals
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        run_in_terminal "python -m clients.client -ID $i -alg $algorithm -run $run"
    done
else
    # Python command failed, handle error
    echo "Error: Data partitioning failed."
fi
