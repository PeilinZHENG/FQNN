## Parameters of GPU
gpu=0

# Parameters of training
data=MNIST
Net=Sig_no_ten_0
input_size=784
embedding_size=100
hidden_size=64
output_size=10
z=None
gradsnorm=2

loss=CE   # CE, BCEWL
opt=Adam
lr=1e-3
wd=0
betas=0.9,0.999
sch=StepLR
gamma=0.5
ss=20

epochs=21
workers=8
batchsize=256
print_freq=20
save_freq=10
seed=0

preNet=Naive_0
checkpointID=_0100

# paths
pretrained="models/${data}/${preNet}/checkpoint${checkpointID}.pth.tar"

source activate
#source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


python ClaTest.py \
  -j $workers -b $batchsize -p $print_freq -s $save_freq --epochs $epochs --data $data --gpu $gpu \
  --loss $loss --opt $opt --lr $lr --wd $wd --betas $betas --sch $sch --gamma $gamma --ss $ss \
  --Net $Net --input_size $input_size --embedding_size $embedding_size \
  --hidden_size $hidden_size --output_size $output_size --z $z \
  --lars --seed $seed --gradsnorm $gradsnorm