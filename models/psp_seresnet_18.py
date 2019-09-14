from keras.optimizers import SGD

from segmentation_models import PSPNet
from segmentation_models.metrics import iou_score, dice_score
from segmentation_models.losses import dice_loss, categorical_crossentropy, binary_crossentropy

model = PSPNet(
    backbone_name='seresnet50',
    input_shape=(144, 912, 3),
    encoder_weights='imagenet',
    activation='sigmoid',
    classes=4,
)
model.compile(
    SGD(lr=0.01, momentum=0.9),
    loss=dice_bce_loss,
    metrics=['accuracy', iou_score, dice_score, binary_crossentropy]
)
