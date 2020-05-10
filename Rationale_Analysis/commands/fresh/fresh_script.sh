bash Rationale_Analysis/commands/model_a_train_script.sh;

export PYTHONPATH=.

bash Rationale_Analysis/commands/saliency_script.sh;

RATIONALE_EXP_NAME=direct \
bash Rationale_Analysis/commands/rationale_and_model_b_script.sh;