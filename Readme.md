1. Train Defect Classifier and observe [class activation map](https://jacobgil.github.io/deeplearning/class-activation-maps)
2. Train (Segmentation Models) on defects and subset of hard negtive examples
3. Ensemble Results
4. Train on Balanced Batch Generator

https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20190017
  Epoch 23/50 loss: 0.1282 - acc: 0.6397 - iou_score: 0.8582 - score: 0.8718 - binary_crossentropy: 0.0702 val_score0.8894179120208278
  model = PSPNet(backbone_name='resnet18', input_shape=(192, 960, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
  model.compile(Adam(lr = 1e-5), loss=dice_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
  random_state=420
  filtered_samples = samples[((samples.score > 0.01) & (samples.has_defect == True)].reset_index(drop=True)


https://www.kaggle.com/nikperi/resnet18-unet/notebook?scriptVersionId=20145646
Epoch 12/50 score-0.8893257091442744
model = PSPNet(backbone_name='mobilenetv2', input_shape=(192, 960, 3), classes=4, activation='sigmoid')
model.compile(Adam(lr = 5e-5), loss=dice_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
filtered_samples = samples[((samples.score > 0.01) & (samples.has_defect == True)].reset_index(drop=True)

https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20197321
Epoch 30/50 score-0.8854644721204584
model = PSPNet(backbone_name='mobilenetv2', input_shape=(192, 960, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(Adam(lr = 1e-5), loss=dice_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
random_state=420
filtered_samples = samples[((samples.score > 0.01) & (samples.has_defect == True)].reset_index(drop=True)

https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20226920
Epoch 30/50 score-0.8917608607899059
model = PSPNet(backbone_name='seresnet18', input_shape=(192, 960, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(Adam(lr = 1e-5), loss=dice_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
random_state=420
filtered_samples = samples[((samples.score > 0.01) & (samples.has_defect == True)].reset_index(drop=True)

https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20235878
Epoch 30/50 score-0.8972400249856891
model = PSPNet(backbone_name='seresnet18', input_shape=(192, 960, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(SGD(lr=0.01, momentum=0.9), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
random_state=420

public score 0.87610
https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20242553
2nd training cycle
Epoch 30/50 score-0.8984038775617426
model = PSPNet(backbone_name='seresnet18', input_shape=(192, 960, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(SGD(lr=0.0001, momentum=0.9), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
random_state=420

public score 0.86724
https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20367341
Epoch 42/50 score-0.8639785990570531
model = PSPNet(backbone_name='seresnet34', input_shape=(192, 1248, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(SGD(lr=0.01, momentum=0.9, decay=0.01), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
random_state=420

https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20379124
Epoch 25/50 0.8731188765470532
model = PSPNet(backbone_name='seresnet18', input_shape=(192, 1248, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(SGD(lr=0.01, momentum=0.9), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
filtered_samples = samples[((samples.score > 0.001) & (samples.has_defect == True)].reset_index(drop=True)

public score 0.87922
https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20399979
Epoch 14 score-0.9017058203414995
model = PSPNet(backbone_name='seresnet18', input_shape=(144, 912, 3), classes=4, encoder_weights='imagenet', activation='sigmoid')
model.compile(SGD(lr=0.01, momentum=0.9), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
filtered_samples = all data

public score 0.88145
https://www.kaggle.com/nikperi/resnet18-unet?scriptVersionId=20409254
2nd training cycle
class_weights = compute_class_weight('balanced', np.unique(samples.iloc[train]['class'].values), samples.iloc[train]['class'].values)[1:]
Epoch 6 score-0.9048350106696693
model = PSPNet(backbone_name='seresnet18', input_shape=(144, 912, 3), classes=4, encoder_weights='imagenet', activation='sigmoid'
model.compile(SGD(lr=0.0001, momentum=0.9), loss=dice_bce_loss, metrics=['accuracy', iou_score, dice_score, binary_crossentropy])
filtered_samples = all data

I think the key to solving this is building a training set the contains all the labeled masks from classes one and two and then selecting some specific subset of no mask data.

If you simply train on class one and two the model will become over sensitive to marking defects and the false positive rate will be very high. using the [test set probing](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106477#latest-621119) provided by @hengck23 the output of this model trained only on class one and two masks is:
```
compare with probed results---
		num_image =  1801(1801)
		num  =  7204(7204)
		neg  =  4898(6172)  0.680
		pos  =  2306(1032)  0.320
		pos1 =  1183( 128)  0.657  0.513
		pos2 =  1123(  43)  0.624  0.487
		pos3 =     0( 741)  0.000  0.000
		pos4 =     0( 120)  0.000  0.000
```

However the false positive rate can be decreased by introducing some training images without a mask.  But it is very important to pick good samples without a mask because there are many samples which are missing masks and will result in the model losing sensitivity.

I suggest using the output of a missing mask model as seen below, combined with some hand selection to pick good negative samples:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2128634%2Fb9b781b9ae8d225d28eae8bbbd1c2dff%2FScreen%20Shot%202019-09-12%20at%2010.26.27%20PM.png?generation=1568341605312124&alt=media)
^ this distribution was generated using a model trained based on [missing masks](https://www.kaggle.com/xhlulu/severstal-predict-missing-masks)

By adding in a few of these samples I achieved the following:
```
compare with probed results---
		num_image =  1801(1801)
		num  =  7204(7204)
		neg  =  4898(6172)  0.680
		pos  =  2306(1032)  0.320
		pos1 =   748( 128)  0.657  0.513
		pos2 =   748(  43)  0.624  0.487
		pos3 =     0( 741)  0.000  0.000
		pos4 =     0( 120)  0.000  0.000
```
^0.9 0.89
public score 0.88645
https://www.kaggle.com/nikperi/resnet18-unet/output?scriptVersionId=20418549

compare with probed results---
		num_image =  1801(1801)
		num  =  7204(7204)
		neg  =  6575(6172)  0.913
		pos  =   629(1032)  0.087
		pos1 =     0( 128)  0.000  0.000
		pos2 =     0(  43)  0.000  0.000
		pos3 =   539( 741)  0.299  0.857
		pos4 =    90( 120)  0.050  0.143

gotta try deeplabv3 https://www.kaggle.com/alexanderliao/deeplabv3
