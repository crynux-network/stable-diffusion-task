import json

from sd_task.task_args import FinetuneLoraTaskArgs, ImageTaskArgs

if __name__ == "__main__":
    dest_file = "./schema/image-task.json"

    schema = ImageTaskArgs.model_json_schema()

    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    dest_file = "./schema/stable-diffusion-finetune-lora-task.json"

    schema = FinetuneLoraTaskArgs.model_json_schema()

    with open(dest_file, mode="w", encoding="utf-8") as f:
        json.dump(schema, f)

    print("json schema output successfully")
