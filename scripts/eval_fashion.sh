ckpt=$1

ZEROSHOT_DATASETS=(
"Fashion200k-CLS"
"Fashion200k-SUBCLS"
"FashionGen-CLS"
"FashionGen-SUBCLS"
"Polyvore-CLS"
)
RETRIEVAL_DATASETS=(
"Fashion200k"
"FashionGen"
"Polyvore"
)

# zero-shot classification
for dataset in "${ZEROSHOT_DATASETS[@]}"; do
    python main.py \
        --name ${ckpt} \
        --imagenet-val ${dataset} \
#
done
# image-text retrieval
for dataset in "${RETRIEVAL_DATASETS[@]}"; do
    python main.py \
        --name ${ckpt} \
        --val-data ${dataset} \
#
done
