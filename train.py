import importutils
from utils.data import clean_training_samples, DataGenerator

if __name__ == '__main__':
if __name__ == '__main__':

    timestamp = time.strftime('%c')
    os.mkdir('models/{}'.format(timestamp))
    os.mkdir('logs/{}'.format(timestamp))
    save_callback = keras.callbacks.ModelCheckpoint(
        'models/' + timestamp + '/weights-{val_binary_crossentropy:.3f}.hdf5',
        monitor='val_binary_crossentropy', verbose=0,
        save_best_only=True, save_weights_only=False,
        mode='auto', period=1
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(timestamp),
        batch_size=BATCH_SIZE,
        update_freq='epoch'
    )
    model = build_model()
    model.fit_generator(
        generator=DataGenerator('data/bdd100k/segmentation/processed_images/train', 'data/bdd100k/segmentation/processed_labels/train'),
        validation_data=DataGenerator('data/bdd100k/segmentation/processed_images/val', 'data/bdd100k/segmentation/processed_labels/val').all(),
        use_multiprocessing=True,
        workers=MAX_CONCURRENCY,
        callbacks=[save_callback, tensorboard_callback],
        epochs=EPOCHS
    )
