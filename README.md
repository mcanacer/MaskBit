# MaskBit from Scratch in JAX/FLAX

This repository contains a from-scratch implementation of the paper:

> ** MaskBit: Embedding-free Image Generation via Bit Tokens **  
> (https://arxiv.org/abs/2409.16211)

You can find the checkpoints these links:
- [tokenizer checkpoint](https://drive.google.com/file/d/1SVry2lMfZ7Y5ru1HvvmFwEkuxLeyNk_4/view?usp=drive_link)
- [maskbit checkpoint](https://drive.google.com/file/d/1_GO0YG94YHHiqI4D2u4iRX5BDUGLmPS9/view?usp=drive_link)

```bash
python train_tokenizer.py configs/tokenizer_14.yaml
```

## 🏁 Training MaskBit

```bash
python train_maskbit.py configs/maskbit_14bit.yaml
```

## 🎨 Inference

```bash
python inference.py configs/maskbit_14bit.yaml
```

## 🖼 Sample Generated Images From CelebA

![Generated Image](gen_images/generated_image0.jpeg)
![Generated Image](gen_images/generated_image1.jpeg)
![Generated Image](gen_images/generated_image2.jpeg)
![Generated Image](gen_images/generated_image3.jpeg)
![Generated Image](gen_images/generated_image4.jpeg)
![Generated Image](gen_images/generated_image5.jpeg)
![Generated Image](gen_images/generated_image6.jpeg)
![Generated Image](gen_images/generated_image7.jpeg)
![Generated Image](gen_images/generated_image8.jpeg)
![Generated Image](gen_images/generated_image9.jpeg)
![Generated Image](gen_images/generated_image10.jpeg)
![Generated Image](gen_images/generated_image11.jpeg)
![Generated Image](gen_images/generated_image12.jpeg)
![Generated Image](gen_images/generated_image13.jpeg)
![Generated Image](gen_images/generated_image14.jpeg)
![Generated Image](gen_images/generated_image15.jpeg)
![Generated Image](gen_images/generated_image16.jpeg)
![Generated Image](gen_images/generated_image17.jpeg)
