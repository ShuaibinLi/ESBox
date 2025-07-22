#!/usr/bin/env bash

# Run examples test
set -e
alias python="/data/lishuaibin/anaconda3/envs/dev_esbox/bin/python"


run_quadratic() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "++ Running quadratic example converge test... ++"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    python ./Quadratic-example/run_local.py --config_file Quadratic-example/local.ymal > temp0.log 2>&1 &&
    
    filename="./temp0.log"

    local ground=0.9999999
    local diff_pos=0.001
    local diff_neg=-0.001
    local done=0
    # for reward in `awk '{print $NF}' $1`
    for reward in `awk '{print $NF}' $filename`
    do
        diff=`awk -v x=$reward -v y=$ground 'BEGIN{print x-y}'`
        k1=`awk -v num1=$diff_neg -v num2=$diff 'BEGIN{print(num1<num2)?"1":"0"}'`
        k2=`awk -v num1=$diff -v num2=$diff_pos 'BEGIN{print(num1<num2)?"1":"0"}'`

        if [ $k1 -eq 1 ] && [ $k2 -eq 1 ]; then
            # echo "Already converge at reward: $reward, ground truth: $ground, diff: $diff"
            done=1
            break
        fi
    done

    rm -rf ./esbox_train_log
    rm -rf $filename
    if [ $done -eq 1 ]; then
        echo 1
    else
        echo 0
    fi
}

run_cartpole() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "++ Running cartpole example converge test ... ++"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    python ./CartPole-example/run_local.py --config_file CartPole-example/local.ymal > temp1.log 2>&1 &&

    filename="./temp1.log"

    local ground=500.0
    local diff_pos=0.01
    local diff_neg=-0.01
    local done=0
    # for reward in `awk '{print $NF}' $1`
    for reward in `awk '{print $NF}' $filename`
    do
        diff=`awk -v x=$reward -v y=$ground 'BEGIN{print x-y}'`
        k1=`awk -v num1=$diff_neg -v num2=$diff 'BEGIN{print(num1<num2)?"1":"0"}'`
        k2=`awk -v num1=$diff -v num2=$diff_pos 'BEGIN{print(num1<num2)?"1":"0"}'`

        if [ $k1 -eq 1 ] && [ $k2 -eq 1 ]; then
            # echo "Already converge at reward: $reward, ground truth: $ground, diff: $diff"
            done=1
            break
        fi
    done

    rm -rf ./esbox_train_log
    rm -rf $filename
    if [ $done -eq 1 ]; then
        echo 1
    else
        echo 0
    fi
}


main() {
    ###
    ret=$(run_quadratic $filename)
    if [ $ret -eq 1 ]; then
        echo "Sucessfully run quadratic example!"
    else
        echo "Failed to run quadratic example!"
    fi

    ###
    ret=$(run_cartpole $filename)
    if [ $ret -eq 1 ]; then
        echo "Sucessfully run cartpole example!"
    else
        echo "Failed to run cartpole example!"
    fi
}

main
