
# UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model [ECCV2024]

## Environment

- Linux
- Python 3.10
- Pytorch 2.2.0
- CUDA 12.1
- transformers 4.39.3
- Pytorch3d 0.7.7 (just for render the results)

## **Demo**
export ckpt_path=./checkpoint/UniTalker-B-[D0-D7].pt 
export test_wav_dir=./example_wav/
export save_path=./results

python -m main.demo --config config/unitalker.yaml \
weight_path ${ckpt_path} \
test_wav_dir ${test_wav_dir} \
test_out_path ${save_path}/out.npz

python -m main.render \
${save_path}/out.npz  ${test_wav_dir} ${save_path}

## **Train**
export save_path=./results
python -m main.train save_path $save_path
