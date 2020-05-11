for search in {0..19};
    do
    EXP_NAME=mu_lambda_search/search_${search} bash Rationale_Analysis/commands/encgen/experiment_script.sh;
    done;