from data_loader import get_datasets
from model import build_model

WAVS_PATH = "./wavs/"
METADATA_PATH = "./metadata.csv"

train_ds, val_ds, char_to_num, _ = get_datasets(METADATA_PATH, WAVS_PATH)
model = build_model(input_dim=193, output_dim=char_to_num.vocabulary_size())
model.summary()

# Start training
model.fit(train_ds, validation_data=val_ds, epochs=50)
model.save("asr_model.h5")
