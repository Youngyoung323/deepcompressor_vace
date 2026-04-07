# INT4 SVDQuant for VACE, then generate for comparison
python -m deepcompressor.app.diffusion.ptq_vace \
    configs/model/wan2.1-vace-14b.yaml \
    configs/svdquant/int4.yaml \
    --skip-eval true \
    --save-model true
