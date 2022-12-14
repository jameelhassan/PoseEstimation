INPLANES = 48   #Edit from 64 for Bottleneck
NUM_FEATS = 84  #Edit from 128 for Bottleneck
EXPANSION = 2
SEPARABLE_ALL = True    # Separable convolutions in all layers of Bottleneck
SEPARABLE_3x3 = True    # Only the 3x3 in the Bottleneck is replaced by separable convs (default True if All Separable)
GHOST = False
CONCAT = True   # Concatenate the output of the residual block with the input instead of adding
PERCEPTUAL_RES = True   # Skip connection from narrowest hourglass point ie: perceptual
PERCEPTUAL_LOSS = True # Perceptual loss between perceptuals of hourglass
PERCEPTUAL_SCALE = 2    # Weighing Scale of perceptual loss
LOSS_WEIGHT = 0.7       # Weight for prediction loss when including perceptual loss

if SEPARABLE_ALL:
    MODEL_TAG = f'Separable convolution at all of Bottlenecks {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}, perceptual loss|Res: {PERCEPTUAL_LOSS}|{PERCEPTUAL_RES}'
elif SEPARABLE_3x3:
    MODEL_TAG = f'Separable convolution at 3x3 Bottleneck only {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}, perceptual loss|Res: {PERCEPTUAL_LOSS}|{PERCEPTUAL_RES}'
elif GHOST:
    MODEL_TAG = f"Ghost Net {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}, perceptual loss|Res: {PERCEPTUAL_LOSS}|{PERCEPTUAL_RES}"
    # print("GHOST net module training running!")
elif CONCAT:
    MODEL_TAG = f"Output concatenation with {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}, perceptual loss|Res: {PERCEPTUAL_LOSS}|{PERCEPTUAL_RES}"
elif PERCEPTUAL_RES:
    MODEL_TAG = f"Perceptual residual connection with {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}, perceptual loss|Res: {PERCEPTUAL_LOSS}|{PERCEPTUAL_RES}"
else:
    MODEL_TAG = None