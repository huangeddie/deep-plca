# PLCA
Probabilistic latent component analysis

### Example usage
```
python main.py --data=mnist --prob --imsize=28 --model=deep-plca --nkern=15 --kern-size=7 --opt=adam --bsz=512 --lr=1e-1 --epochs=1 --recon=ce --beta1=0 --beta2=2 --beta3=0
```
Runs deep PLCA on MNIST, whose images have been converted to probability distributions. It specifies 15 components whose features are 7x7. It uses the Adam optimizer with batch size 512, learning rate 0.1, and 1 epoch. The reconstruction loss is set to cross entropy, with prior, impulse and kernel entropy weight losses set to 0, 2, 0 respectively.

See the `main.py` file to see the full list of arguments and their descriptions.
