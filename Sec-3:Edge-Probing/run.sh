#!/usr/bin/env bash
ontonotes_dir=/work/LM-embryology
model_dir=/work/LM-embryology/pytorch_model
config_and_vocab=/work/LM-embryology/LM

function run_POS () {
  step=$1
  layer=$2
  random_num=$3
  model=$4
  mkdir -p tmp$random_num
  for data in train test development; do  
    python3 preprocess/extract_feature.py \
      -i $ontonotes_dir/ontonotes/const/pos/$data.json \
      -m $model_dir/$model/pytorch_model_$step.bin \
      -o ./tmp$random_num/ \
      -c $config_and_vocab/$model/config.json \
      -t $config_and_vocab/$model/vocab.model \
      -l $ontonotes_dir/ontonotes/const/pos/labels.txt \
      -p POS \
      --data_type $data \
      --layer $layer
  done  
  python3 main.py \
    -t tmp$random_num/POS-train-$model-$step.pkl \
    -d tmp$random_num/POS-development-$model-$step.pkl \
    -g tmp$random_num/POS-test-$model-$step.pkl \
    -c 48 \
    -s 30000 \
    --pretrain_step $step \
    --o result/$model/pos/$layer.txt
  rm -rf tmp$random_num
}

function run_SRL () {
  step=$1
  layer=$2
  random_num=$3
  model=$4
 mkdir -p tmp$random_num
  for data in train test development; do  
    python3 preprocess/extract_feature.py \
      -i $ontonotes_dir/ontonotes/srl/$data.json \
      -m $model_dir/$model/pytorch_model_$step.bin \
      -o ./tmp$random_num/ \
      -c $config_and_vocab/$model/config.json \
      -t $config_and_vocab/$model/vocab.model \
      -l $ontonotes_dir/ontonotes/srl/labels.txt \
      -p SRL \
      --data_type $data \
      --layer $layer
  done  
  python3 main.py \
    -t tmp$random_num/SRL-train-$model-$step.pkl \
    -d tmp$random_num/SRL-development-$model-$step.pkl \
    -g tmp$random_num/SRL-test-$model-$step.pkl \
    -c 66 \
    -s 30000 \
    --pretrain_step $step \
    --o result/$model/srl/$layer.txt \
    --span2
  rm -rf tmp$random_num
}

function run_coref () {
  step=$1
  layer=$2
  random_num=$3
  model=$4
  mkdir -p tmp$random_num
  for data in train test development; do  
    python3 preprocess/extract_feature.py \
      -i $ontonotes_dir/ontonotes/coref/$data.json \
      -m $model_dir/$model/pytorch_model_$step.bin \
      -o ./tmp$random_num/ \
      -c $config_and_vocab/$model/config.json \
      -t $config_and_vocab/$model/vocab.model \
      -l $ontonotes_dir/ontonotes/coref/labels.txt \
      -p coref \
      --data_type $data \
      --layer $layer
  done  
  python3 main.py \
    -t tmp$random_num/coref-train-$model-$step.pkl \
    -d tmp$random_num/coref-development-$model-$step.pkl \
    -g tmp$random_num/coref-test-$model-$step.pkl \
    -c 2 \
    -s 30000 \
    --pretrain_step $step \
    --o result/$model/coref/$layer.txt \
    --span2
  rm -rf tmp$random_num
}

function run_const () {
  step=$1
  layer=$2
  random_num=$3
  model=$4
  mkdir -p tmp$random_num
  for data in train test development; do  
    python3 preprocess/extract_feature.py \
      -i $ontonotes_dir/ontonotes/const/nonterminal/$data.json \
      -m $model_dir/$model/pytorch_model_$step.bin \
      -o ./tmp$random_num/ \
      -c $config_and_vocab/$model/config.json \
      -t $config_and_vocab/$model/vocab.model \
      -l $ontonotes_dir/ontonotes/const/nonterminal/labels.txt \
      -p const \
      --data_type $data \
      --layer $layer
  done  
  python3 main.py \
    -t tmp$random_num/const-train-$model-$step.pkl \
    -d tmp$random_num/const-development-$model-$step.pkl \
    -g tmp$random_num/const-test-$model-$step.pkl \
    -c 30 \
    -s 30000 \
    --pretrain_step $step \
    --o result/$model/const/$layer.txt
  rm -rf tmp$random_num
}


begin=$1
end=$2
random_num=$3
for (( layer=$begin; layer<=$end; layer+=4 )); do
  for step in {250000..500000..40000} 500000; do
    run_const $step $layer $random_num albert
    #run_coref $step $layer $random_num albert
    #run_POS $step $layer $random_num bert
    run_SRL $step $layer $random_num albert
  done
done
