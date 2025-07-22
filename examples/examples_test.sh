#!/usr/bin/env bash

# Run examples test
set -e
alias python="/data/lishuaibin/anaconda3/envs/dev_esbox/bin/python"


sed_max_runs() {
    for filename in ./tuned_configs_cp/*
    do
        sed -i "s/'max_runs': 200/'max_runs': 2/g" ${filename}
        sed -i "s/'eval_every_run': 10/'eval_every_run': 1/g" ${filename}
        # sed -i "s/'ray_addr': ''/'ray_addr': ''/g" ${filename}
        # sed -i "/'num_workers'/d" ${filename}

        # echo $filename
    done

}

run_func_test() {
    for alg in openaies ars cmaes sep-cmaes
    do 
        python run_function.py --config_file ./tuned_configs_cp/${alg}_function.ymal
        python run_function_model.py --config_file ./tuned_configs_cp/${alg}_function_model.ymal
        echo "++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++ func test with ${alg} test done ++++++++"
        echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    done
}

run_mujoco_test() {
    for alg in openaies ars nsraes cmaes sep-cmaes
    do 
        python run_mujoco.py --config_file ./tuned_configs_cp/${alg}_mujoco.ymal
        sleep 60s
        echo "++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "+++++++ mujoco test with ${alg} done ++++++++"
        echo "++++++++++++++++++++++++++++++++++++++++++++++++"
    done
}


main() {
    cp -R ./tuned_configs ./tuned_configs_cp
    rm -rf ./tuned_configs_cp/README.md

    sed_max_runs

    run_func_test

    ray start --head --num-cpus 50 --port 6379
    run_mujoco_test

    if [$? -ne 0]; then 
        echo "examples test failed."
    else
        echo "examples test success"
    fi
    ray stop
    rm -rf ./tuned_configs_cp
    rm -rf ./esbox_train_log
}

main
