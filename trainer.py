from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import yaml
import pickle
import os
import shutil
import pickle

from learner.net import InverseNet
from learner.data import train, val

EARLY_STOPPING = True


patience = 30
max_epochs = 400


with open('learner/config.yaml') as file:
    config = yaml.safe_load(file.read())

config['trainer'] = {'patience' : patience, 'max_epochs' : max_epochs}

time = datetime.datetime.now().isoformat(':',timespec='minutes')

if not os.path.isfile('weight/{}'.format(time)):
    os.makedirs('weight/{}'.format(time))
shutil.copy('learner/config.yaml', 'weight/{}/config.yaml'.format(time))

with open('weight/{}/config.yaml'.format(time), 'wb') as f:
    yaml.dump(config, f, encoding='utf-8', allow_unicode=True)

model_checkpoint = ModelCheckpoint(
    "weight/",
    filename = "{}/weight".format(time),
    monitor = "val_loss",
    mode = "min",
    save_top_k = 1,
    save_last = False,
)

early_stopping = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = patience,
)

Net = InverseNet()

if EARLY_STOPPING:
    trainer = Trainer(callbacks = [model_checkpoint, early_stopping], max_epochs = max_epochs)
else:
    trainer = Trainer(callbacks = [model_checkpoint], max_epochs = max_epochs, strategy = 'ddp')

save_train = open('weight/{}/train.txt'.format(time), 'wb')
save_val = open('weight/{}/valt.txt'.format(time), 'wb')
pickle.dump(train, save_train)
pickle.dump(val, save_val)

save_train = open('weight/{}/train.txt'.format(time), 'wb')
save_val = open('weight/{}/valt.txt'.format(time), 'wb')
pickle.dump(train, save_train)
pickle.dump(val, save_val)

trainer.fit(Net)
print(trainer.callback_metrics)