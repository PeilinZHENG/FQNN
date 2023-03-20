## Parameters of GPU
gpu=0
threads=16

# Parameters of training
L=12
data=Ins_12
adj=None
Net=Naive_test
input_size=$(($L*$L))
embedding_size=100
hidden_size=64
output_size=2
z=1e-3j
restr=False       # False: fc, 1: 1D NN, 2: 2D NN, 3: 1D NNN, 4: 2D NNN
diago=False       # True, False, 1: 1D NN, 2: 2D NN, 3: 1D NNN, 4: 2D NNN
hermi=True        # True, False, 0: naive hermi
bound=1         # initial bound

entanglement=False   # False, int or float
delta=0
tc=None
gradsnorm=False

loss=CE   # NLL, CE, BCE, BCEWL
opt=Adam
lr=1e-3
wd=0
betas=0.9,0.999
sch=StepLR
gamma=0.5
ss=20
drop=0
disor=0

epochs=21
workers=8
batchsize=256
print_freq=20
save_freq=10
seed=0

category=STRUCTURE
preNet=Naive_h_4-
checkpointID=checkpoint_0100

# paths
pretrained="models/${data}/${category}/${preNet}/model_best.pth.tar"
resume="models/${data}/${category}/${preNet}/${checkpointID}.pth.tar"

source activate
#source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


python Train.py \
  -t $threads -j $workers -b $batchsize -p $print_freq -s $save_freq --epochs $epochs --data $data --adj $adj \
  --gpu $gpu --loss $loss --opt $opt --lr $lr --wd $wd --betas $betas --sch $sch --gamma $gamma --ss $ss \
  --Net $Net --z $z --entanglement $entanglement --delta $delta --tc $tc --gradsnorm $gradsnorm --seed $seed \
  --input_size $input_size --embedding_size $embedding_size --hidden_size $hidden_size --output_size $output_size \
  --drop $drop --disor $disor --init_bound $bound --restr $restr --hermi $hermi --diago $diago \
  --scale --lars
