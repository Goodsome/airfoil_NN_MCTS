from GAN.cgan_cp import *

model = CGAN()
model.load_models()
print(model.predict(0, one_cl_cd=True))
