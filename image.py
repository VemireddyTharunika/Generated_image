  
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


# Model and tokenizer loading
model_id = "vikhyatk/moondream2"
revision = "2024-03-06"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Image loading
image_path = 'demp.png'
image = Image.open(image_path)

# Display the image
image.show()

# Encoding the image
enc_image = model.encode_image(image)

# Asking the model to describe the image
description = model.answer_question(enc_image, "Describe this image.", tokenizer)
print("Generated Description:", description)
