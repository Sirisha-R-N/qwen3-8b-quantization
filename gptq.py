from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "qwen3-8b" # replace with your model path
quant_path = "gptq-4" # replace with the destination path

calibration_dataset = load_dataset(
    "imdb",
    split="train"
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128) # for 8 bit, let bits=8

model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)
