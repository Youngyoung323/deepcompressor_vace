# Reference mode — inference only, save reference videos
python -m deepcompressor.app.diffusion.ptq_vace \
    configs/model/wan2.1-vace-14b.yaml \
    --output-dirname reference \
    --skip-eval true
