ckpt=$1

ZEROSHOT_DATASETS=(
"RSICD-CLS"
"UCM-CLS"
"WHU-RS19"
"RSSCN7"
"AID"
)
RETRIEVAL_DATASETS=(
"RSICD"
"UCM"
"Sydney"
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
