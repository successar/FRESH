bash Rationale_Analysis/commands/model_a_train_script.sh;

export PYTHONPATH=.

bash Rationale_Analysis/commands/fresh/saliency_script.sh;

THRESHOLDER_EXP_NAME=$MAX_LENGTH_RATIO \
bash Rationale_Analysis/commands/fresh/thresholder_and_model_b_script.sh;