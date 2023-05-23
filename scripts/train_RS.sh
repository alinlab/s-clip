METHOD=(
"base"
"ours"
)
SEED=("0")

for method in "${METHOD[@]}"; do
for seed in "${SEED[@]}"; do
    torchrun --nproc_per_node 4 -m main \
        --model "RN50" \
        --pretrained openai \
        --train-data "RS-ALL" \
        --label-ratio "0.1" \
        --val-data "RS-ALL" \
        --imagenet-val "RSICD-CLS" \
        --keyword-path "keywords/RS/class-name.txt" \
        --lr 5e-5 \
        --batch-size 64 \
        --warmup 10 \
        --epochs 25 \
        --zeroshot-frequency 5 \
        --precision amp \
        --method "${method}" \
        --seed "${seed}" \
        --report-to wandb \
        --wandb-project-name "S-CLIP" \
#
done
done
