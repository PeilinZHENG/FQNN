## Parameters of GPU
gpu=2
threads=8

# Parameters of DMFT
count=20
iota=0
momentum=0.5
maxEpoch=100
filling=0.5
tol_sc=1e-6
tol_bi=1e-6

# Parameters of training
L=14
data=FK_${L}
Net=Naive_1
input_size=$(($L*$L))
embedding_size=100
hidden_size=64
output_size=2
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

epochs=10
workers=8
batchsize=100
print_freq=10
save_freq=1
seed=0


preNet=Naive_h_4-
checkpointID=checkpoint_0100

# paths
pretrained="models/${data}/${preNet}/model_best.pth.tar"
resume="models/${data}/${preNet}/${checkpointID}.pth.tar"


source activate
#source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


python FK_Train.py \
  -t $threads -j $workers -b $batchsize -p $print_freq -s $save_freq --epochs $epochs --gpu $gpu --count $count \
  --iota $iota --momentum $momentum --maxEpoch $maxEpoch --filling $filling --tol_sc $tol_sc --tol_bi $tol_bi\
  --data $data --opt $opt --loss $loss --lr $lr --wd $wd --betas $betas --sch $sch --gamma $gamma --ss $ss \
  --Net $Net --entanglement $entanglement --delta $delta --tc $tc --gradsnorm $gradsnorm --seed $seed \
  --input_size $input_size --embedding_size $embedding_size --hidden_size $hidden_size --output_size $output_size \
  --drop $drop --disor $disor --init_bound $bound --restr $restr --hermi $hermi --diago $diago --double --scale \

