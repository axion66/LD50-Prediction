task=classification     
dataset=hiv

radius=1
dim=64
layer_hidden=6
layer_output=6

batch_train=64
batch_test=16
lr=2e-4
lr_decay=0.9999
decay_interval=10
iteration=50

setting="${dataset}--radius${radius}--dim${dim}--layer_hidden${layer_hidden}--layer_output${layer_output}--batch_train${batch_train}--batch_test${batch_test}--lr${lr}--lr_decay${lr_decay}--decay_interval${decay_interval}--iteration${iteration}"

python3 train.py $task $dataset $radius $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $iteration $setting