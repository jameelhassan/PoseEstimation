INPLANES = 64   #Edit from 64
NUM_FEATS = 128  #Edit from 128
EXPANSION = 2
SEPARABLE_ALL = False
SEPARABLE_3x3 = False
GHOST = False
CONCAT = False
LOSS_WEIGHT = 0.8

if SEPARABLE_ALL:
    MODEL_TAG = f'Separable convolution at all of Bottlenecks {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}'
elif SEPARABLE_3x3:
    MODEL_TAG = f'Separable convolution at 3x3 Bottleneck only {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}'
elif GHOST:
    MODEL_TAG = f"Ghost Net {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}"
    print("GHOST net module training running!")
elif CONCAT:
    MODEL_TAG = f"Output concatenation with {INPLANES}-{NUM_FEATS} filters, expansion {EXPANSION}"
else:
    MODEL_TAG = None